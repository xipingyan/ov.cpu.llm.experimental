from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils, save_model
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from .utils import show_model, make_mha, make_fc, make_combined_fc, pt_as_np, make_rms_norm, make_embedding, save_tokenzier, OV_XML_FILE_NAME, configs as make_configs, swish
from tqdm import tqdm
import nncf
from nncf.parameters import CompressWeightsMode

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'model.layers.self_attn'
    # layerNorm operation
    input_layernorm = make_rms_norm('model.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs['rms_norm_eps'], name_suffix)

    if not make_configs["fuse_qkv"]:
        q = make_fc('model.layers.self_attn.q_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
        k = make_fc('model.layers.self_attn.k_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
        v = make_fc('model.layers.self_attn.v_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
        inputs = [q, k, v]
    else:
        qkv = make_combined_fc(['model.layers.self_attn.q_proj', 'model.layers.self_attn.k_proj', 'model.layers.self_attn.v_proj'], input_layernorm, consts['layers'][layer_idx], name_suffix)
        inputs = [qkv]

    # custom op
    attn_output = make_mha(inputs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}')

    attn_output = make_fc('model.layers.self_attn.o_proj', attn_output, consts['layers'][layer_idx], name_suffix)

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast='numpy', name=f'{name_prefix}.add0{name_suffix}')
    post_attention_layernorm = make_rms_norm('model.layers.post_attention_layernorm', attn_output, consts['layers'][layer_idx], configs['rms_norm_eps'], name_suffix)

    # mlp
    def mlp(states):
        gate_proj = make_fc('model.layers.mlp.gate_proj', states, consts['layers'][layer_idx], name_suffix)
        silu = swish(gate_proj, name=f'{name_prefix}.mlp.silu{name_suffix}')
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
    # [2 * n_layers, max_kv_len, batch, n_head, head_size]
    kv_cache = opset.parameter([2 * configs['layer_num'], -1, -1, configs['head_num'], configs['head_size']], Type.f32, name='kv_cache')
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name='beam_table')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f32, name='attn_mask')
    # [max_kv_len, rotary_dims]
    cos_tab = opset.parameter([-1, configs['rotary_dims']], Type.f32, name='cos_tab')
    sin_tab = opset.parameter([-1, configs['rotary_dims']], Type.f32, name='sin_tab')

    inputs_embeds = make_embedding('model.embed_tokens.weight', input_ids, consts)
    hidden_states = inputs_embeds

    for i in tqdm(range(configs['layer_num'])):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)

    # only keep the last token
    hidden_states = opset.slice(hidden_states, np.array([-1]), np.array([np.iinfo(np.int32).max]), np.array([1]), np.array([1]))

    # final_layernorm
    final_layernorm = make_rms_norm('model.norm', hidden_states, consts, configs['rms_norm_eps'])
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
    assert(model.config.use_sliding_window == False)
    assert(model.config.rope_theta == 1000000.0)

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
    return configs, consts, model.config

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, default='Model ID (can be a Hugginface Hub id, or a local directory)')
    parser.add_argument('--ov_model_path', type=str, nargs='?', default='./gen/qwen2-7b-chat/')
    parser.add_argument('--quant_type', type=str, nargs='?', default='', choices=['','f16','nncf_w8', 'INT8_ASYM', 'INT8_SYM'])
    parser.add_argument("--fuse-qkv", action="store_true", help="fuse Q/K/V Linear projections into single Linear")
    args = parser.parse_args()
    quant_f16 = False

    if args.quant_type:
        args.ov_model_path = os.path.join(args.ov_model_path, args.quant_type)

    os.makedirs(args.ov_model_path, exist_ok=True)

    quant_type = args.quant_type
    if quant_type == 'nncf_w8':
        make_configs['quant_type'] = quant_type
    else:
        make_configs['quant_type'] = ''

    make_configs["fuse_qkv"] = args.fuse_qkv

    print(f'make_configs:')
    for k, v in make_configs.items():
        print(f'	{k}: {v}')

    configs, consts, org_config = get_params_from_model(args.org_model_path)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    beg = time.time()

    # https://github.com/openvinotoolkit/nncf/blob/6f1b2dd82bf97991e33116e63c4299fcdaf35060/nncf/quantization/quantize_model.py#L362
    # https://github.com/openvinotoolkit/nncf/blob/6f1b2dd82bf97991e33116e63c4299fcdaf35060/nncf/parameters.py#L67
    #  quant_type can be any string in CompressWeightsMode: 
    #       INT8_SYM / INT8_ASYM / INT4_SYM / INT4_ASYM / NF4 / E2M1
    for i in  CompressWeightsMode:
        if i.name == quant_type or i.value == quant_type:
            model = nncf.compress_weights(model,
                                          mode = eval(f"CompressWeightsMode.{i.name}"),
                                          ratio = 1.0,
                                          group_size = -1)
            break

    save_model(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME), quant_type == 'f16')
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
    print(f'save tokenzier to "{args.ov_model_path}" ...')
    save_tokenzier(args.org_model_path, args.ov_model_path)
    # save original json
    org_config.to_json_file(f'{args.ov_model_path}/config_org.json')
