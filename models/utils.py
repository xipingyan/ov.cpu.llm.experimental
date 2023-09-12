from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from openvino.runtime.op import Constant
import numpy as np

ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
custom_opset = opset_utils._get_node_factory()
custom_opset.add_extension(ext_path)

def pt_as_np(t):
    if t is not None: return t.detach().numpy().astype(np.float32)
    return None

def show_model(m):
    print('inputs of the model:')
    for port, _input in enumerate(m.inputs):
        print('	[{}] {}'.format(port, _input))
    print('outputs of the model:')
    for port, _output in enumerate(m.outputs):
        print('	[{}] {}'.format(port, _output))

def make_mha(qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
             layer_idx, rotary_dim, n_hidden, n_head, name, num_kv_heads=0, rope_type='modified', multi_query_is_planar=False):
    qkvs_len = len(qkvs)
    mha_attr = {'arg_kv_cache': qkvs_len,
                'arg_beam_table': qkvs_len + 1,
                'arg_attn_mask': qkvs_len + 2,
                'arg_cos': qkvs_len + 3,
                'arg_sin': qkvs_len + 4,
                'layer_id': layer_idx,
                'rotary_dims': rotary_dim,
                'n_hidden': n_hidden,
                'n_head': n_head,
                'num_kv_heads': num_kv_heads,
                'multi_query_is_planar': multi_query_is_planar,
                'rope_type': ['original', 'modified'].index(rope_type)}

    if qkvs_len == 1:
        mha_attr['arg_q'] = 0
        mha_attr['arg_k'] = 0
        mha_attr['arg_v'] = 0
    else:
        mha_attr['arg_q'] = 0
        mha_attr['arg_k'] = 1
        mha_attr['arg_v'] = 2

    output = custom_opset.create('MultiHeadAttention', 
        [*qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab], mha_attr)
    output.set_friendly_name(name)
    return output

def make_fc(key, input, consts, name_suffix=''):
    weights = Constant(consts[f'{key}.weight'], True)
    weights.set_friendly_name(name=f'{key}.weight{name_suffix}')
    matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f'{key}.matmul{name_suffix}')
    if consts[f'{key}.bias'] is not None:
        bias = Constant(consts[f'{key}.bias'], True)
        bias.set_friendly_name(name=f'{key}.bias{name_suffix}')
        matmul = opset.add(matmul, bias, auto_broadcast='numpy', name=f'{key}.add{name_suffix}')
    return matmul

def make_mvn(key, input, consts, configs, name_suffix=''):
    mvn = opset.mvn(input, axes=[-1], normalize_variance=True, eps=configs['layer_norm_eps'], eps_mode="inside_sqrt", name=f'{key}.mvn{name_suffix}')
    if consts[f'{key}.weight'] is not None:
        weights = opset.constant(consts[f'{key}.weight'], Type.f32, name=f'{key}.weight{name_suffix}')
        mvn = opset.multiply(mvn, weights, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    if consts[f'{key}.bias'] is not None:
        bias = opset.constant(consts[f'{key}.bias'], Type.f32, name=f'{key}.bias{name_suffix}')
        mvn = opset.add(mvn, bias, auto_broadcast='numpy', name=f'{key}.add{name_suffix}')
    return mvn

def make_rms_norm(key, input, consts, epsilon, name_suffix=''):
    weights = opset.constant(consts[f'{key}.weight'], Type.f32, name=f'{key}.weight{name_suffix}')
    pow = opset.multiply(input, input, name=f'{key}.pow{name_suffix}')
    #pow = opset.power(input, np.array([2], np.float32), name=f'{key}.pow{name_suffix}')
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f'{key}.var{name_suffix}')
    add = opset.add(variance, opset.constant(epsilon, Type.f32), name=f'{key}.add{name_suffix}')
    sqrt = opset.sqrt(add, name=f'{key}.sqrt{name_suffix}')
    div = opset.divide(input, sqrt, name=f'{key}.div{name_suffix}')
    mul = opset.multiply(div, weights, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    return mul
