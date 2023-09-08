from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from utils import show_model, make_mha, make_fc, pt_as_np

def make_rms_norm(key, input, consts, configs, name_suffix=''):
    weights = opset.constant(consts[f'{key}.weight'], Type.f32, name=f'{key}.weight{name_suffix}')
    pow = opset.multiply(input, input, name=f'{key}.pow{name_suffix}')
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f'{key}.var{name_suffix}')
    add = opset.add(variance, opset.constant(configs['rms_norm_eps'], Type.f32), name=f'{key}.add{name_suffix}')
    sqrt = opset.sqrt(add, name=f'{key}.sqrt{name_suffix}')
    div = opset.divide(input, sqrt, name=f'{key}.div{name_suffix}')
    mul = opset.multiply(div, weights, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    return mul

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'model.layers.self_attn'
    # layerNorm operation
    input_layernorm = make_rms_norm('model.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs, name_suffix)

    q = make_fc('model.layers.self_attn.q_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    k = make_fc('model.layers.self_attn.k_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    v = make_fc('model.layers.self_attn.v_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)

    # custom op
    attn_output = make_mha([q, k, v], kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}')

    attn_output = make_fc('model.layers.self_attn.o_proj', attn_output, consts['layers'][layer_idx], name_suffix)

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast='numpy', name=f'{name_prefix}.add0{name_suffix}')
    post_attention_layernorm = make_rms_norm('model.layers.post_attention_layernorm', attn_output, consts['layers'][layer_idx], configs, name_suffix)

    # mlp
    def mlp(states):
        gate_proj = make_fc('model.layers.mlp.gate_proj', states, consts['layers'][layer_idx], name_suffix)
        silu = opset.swish(gate_proj, name=f'{name_prefix}.mlp.silu{name_suffix}')
        up_proj = make_fc('model.layers.mlp.up_proj', states, consts['layers'][layer_idx], name_suffix)
        mul = opset.multiply(silu, up_proj, auto_broadcast='numpy', name=f'{name_prefix}.mlp.mul{name_suffix}')
        down_proj = make_fc('model.layers.mlp.down_proj', mul, consts['layers'][layer_idx], name_suffix)
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast='numpy', name=f'{name_prefix}.add1{name_suffix}')
    return output

def create_model(configs, consts):
    print(f'start generate ov model...')
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    # [2 * n_layers, batch, n_head, max_kv_len, head_size]
    kv_cache = opset.parameter([2 * configs['layer_num'], -1, configs['head_num'], -1, configs['head_size']], Type.f32, name='kv_cache')
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name='beam_table')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f32, name='attn_mask')
    # [max_kv_len, rotary_dims//2]
    cos_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='cos_tab')
    sin_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='sin_tab')

    key = 'model.embed_tokens.weight'
    embed_in_const = opset.constant(consts[key], Type.f32, name=key)
    inputs_embeds = opset.gather(embed_in_const, indices=input_ids, axis=0)
    hidden_states = inputs_embeds

    for i in range(configs['layer_num']):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    final_layernorm = make_rms_norm('model.norm', hidden_states, consts, configs)
    # embed_out
    embed_out = make_fc('lm_head', final_layernorm, consts)
    embed_out_result = opset.result(embed_out, name='logits')
    cost = time.time() - beg
    print(f'generate ov model done, cost {cost:.2f} seconds.')
    return Model([embed_out_result],
                 [input_ids, kv_cache, beam_table, attn_mask, cos_tab, sin_tab])

def get_params_from_model(path):
    print(f'extracting from model "{path}"...')
    beg = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cpu').eval()

    assert(model.config.num_key_value_heads == model.config.num_attention_heads)
    assert(model.config.hidden_act in ['silu'])
    assert(model.config.rope_scaling is None)

    configs = {
        'layer_num': model.config.num_hidden_layers,
        'head_num': model.config.num_attention_heads,
        'head_size': model.config.hidden_size // model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'rotary_dims': int(model.config.hidden_size // model.config.num_attention_heads),
        #'gelu_mode': model.config.hidden_act,
        #'intermediate_size': model.config.intermediate_size,
        #'num_key_value_heads': model.config.num_key_value_heads,
        'rms_norm_eps': model.config.rms_norm_eps,
    }

    consts = {
        'model.embed_tokens.weight': pt_as_np(model.model.embed_tokens.weight),
        'model.norm.weight': pt_as_np(model.model.norm.weight),
        'lm_head.weight': pt_as_np(model.lm_head.weight),
        'lm_head.bias': pt_as_np(model.lm_head.bias),
        'layers': [
            {
                'model.layers.input_layernorm.weight': pt_as_np(l.input_layernorm.weight),
                'model.layers.post_attention_layernorm.weight': pt_as_np(l.post_attention_layernorm.weight),
                'model.layers.self_attn.q_proj.bias': pt_as_np(l.self_attn.q_proj.bias),
                'model.layers.self_attn.q_proj.weight': pt_as_np(l.self_attn.q_proj.weight),
                'model.layers.self_attn.k_proj.bias': pt_as_np(l.self_attn.k_proj.bias),
                'model.layers.self_attn.k_proj.weight': pt_as_np(l.self_attn.k_proj.weight),
                'model.layers.self_attn.v_proj.bias': pt_as_np(l.self_attn.v_proj.bias),
                'model.layers.self_attn.v_proj.weight': pt_as_np(l.self_attn.v_proj.weight),
                'model.layers.self_attn.o_proj.bias': pt_as_np(l.self_attn.o_proj.bias),
                'model.layers.self_attn.o_proj.weight': pt_as_np(l.self_attn.o_proj.weight),
                'model.layers.mlp.gate_proj.bias': pt_as_np(l.mlp.gate_proj.bias),
                'model.layers.mlp.gate_proj.weight': pt_as_np(l.mlp.gate_proj.weight),
                'model.layers.mlp.up_proj.bias': pt_as_np(l.mlp.up_proj.bias),
                'model.layers.mlp.up_proj.weight': pt_as_np(l.mlp.up_proj.weight),
                'model.layers.mlp.down_proj.bias': pt_as_np(l.mlp.down_proj.bias),
                'model.layers.mlp.down_proj.weight': pt_as_np(l.mlp.down_proj.weight)
            } for l in model.model.layers
        ],
    }
    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('org_model_path', type=str, nargs='?', default='/home/llm_irs/pytorch_frontend_models/llama-2-7b-chat/pytorch_original/')
    parser.add_argument('ov_model_path', type=str, nargs='?', default='./gen/llama-2-7b-chat.xml')
    args = parser.parse_args()

    configs, consts = get_params_from_model(args.org_model_path)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    beg = time.time()
    serialize(model, args.ov_model_path)
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
