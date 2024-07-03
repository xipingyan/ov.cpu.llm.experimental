from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset13 as opset
import numpy as np
import sys, os
import argparse
import time
from utils import show_model, make_mha, make_fc, make_mvn, make_embedding, save_tokenzier, OV_XML_FILE_NAME, configs as make_configs
from tqdm import tqdm

def layer(configs, consts, layer_idx, hidden_states, attn_mask):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'transformer.h'
    # layerNorm operation
    input_layernorm = make_mvn('transformer.h.ln_1', hidden_states, consts['layers'][layer_idx], configs, name_suffix)

    weight = consts['layers'][layer_idx]['transformer.h.attn.c_attn.weight']
    new_weight_oc = weight.shape[0] // 3
    bias = consts['layers'][layer_idx]['transformer.h.attn.c_attn.bias']
    q = make_fc('transformer.h.attn.q_proj', input_layernorm, 
                {
                    'transformer.h.attn.q_proj.bias': bias[:new_weight_oc],
                    'transformer.h.attn.q_proj.weight': weight[:new_weight_oc,]
                },
                name_suffix)
    k = make_fc('transformer.h.attn.k_proj', input_layernorm,
                {
                    'transformer.h.attn.k_proj.bias': bias[new_weight_oc : new_weight_oc * 2],
                    'transformer.h.attn.k_proj.weight': weight[new_weight_oc : new_weight_oc * 2,]
                },
                name_suffix)
    v = make_fc('transformer.h.attn.v_proj', input_layernorm,
                {
                    'transformer.h.attn.v_proj.bias': bias[new_weight_oc * 2:],
                    'transformer.h.attn.v_proj.weight': weight[new_weight_oc * 2:,]
                },
                name_suffix)

    attn_output = opset.scaled_dot_product_attention(q, k, v, attn_mask, name=f'{name_prefix}.sdpa{name_suffix}')

    attn_output = make_fc('transformer.h.attn.c_proj', attn_output, consts['layers'][layer_idx], name_suffix)

    # mlp
    def mlp(states):
        c_fc = make_fc('transformer.h.mlp.c_fc', states, consts['layers'][layer_idx], name_suffix)
        gelu = opset.gelu(c_fc, approximation_mode=configs['gelu_mode'], name=f'{name_prefix}.mlp.gelu{name_suffix}')
        c_proj = make_fc('transformer.h.mlp.c_proj', gelu, consts['layers'][layer_idx], name_suffix)
        return c_proj

    # residual connection.
    hidden_states = opset.add(attn_output, hidden_states, auto_broadcast="numpy", name=f'{name_prefix}.add0{name_suffix}')
    mlp_output = mlp(make_mvn('transformer.h.ln_2', hidden_states, consts['layers'][layer_idx], configs, name_suffix))
    output = opset.add(hidden_states, mlp_output, auto_broadcast="numpy", name=f'{name_prefix}.add1{name_suffix}')
    return output

def create_model(configs, consts):
    print(f'start generate ov model...')
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.boolean, name='attention_mask')
    attn_mask_reshape = opset.unsqueeze(attn_mask, [1, 2], name='attention_mask_reshape')

    inputs_embeds = make_embedding('transformer.wte.weight', input_ids, consts)
    # transformers/models/gpt2/modeling_gpt2.py
    # past_length = 0
    # position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    shape_of_input_ids = opset.shape_of(input_ids, Type.i32, name='shape_of_input_ids')
    gather_seq_len = opset.gather(shape_of_input_ids, 1, 0, name='gather_seq_len')
    range_pos = opset.range(0, gather_seq_len, 1, Type.i32, name='range_pos')
    range_pos_2d = opset.unsqueeze(range_pos, 0, name='range_pos_2d')
    position_embeds = make_embedding('transformer.wpe.weight', range_pos_2d, consts)

    hidden_states = inputs_embeds + position_embeds

    for i in tqdm(range(configs['layer_num'])):
        hidden_states = layer(configs, consts, i, hidden_states, attn_mask_reshape)
    # final_layernorm
    final_layernorm = make_mvn('transformer.ln_f', hidden_states, consts, configs)
    # embed_out
    if 'lm_head.weight' in consts:
        embed_out = make_fc('lm_head', final_layernorm, consts)
    else:
        embed_out = make_fc('score', final_layernorm, consts)
    embed_out.output(0).get_tensor().set_names({'logits'})
    embed_out_result = opset.result(embed_out, name='logits')
    cost = time.time() - beg
    print(f'generate ov model done, cost {cost:.2f} seconds.')
    return Model([embed_out_result],
                 [input_ids, attn_mask])

def get_params_from_model(path, is_cls):
    beg = time.time()
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
    if is_cls:
        print(f'extracting from classification model "{path}"...')
        config = AutoConfig.from_pretrained(path, num_labels=2, max_position_embeddings=1024)
        config.pad_token_id = config.eos_token_id
        # create a fake model w/o pre-trained para from config
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        print(f'extracting from gen model "{path}"...')
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cpu').eval()
    print(model)
    assert(model.config.activation_function in ['gelu_new', 'gelu'])

    configs = {
        'layer_num': model.config.n_layer,
        'head_num': model.config.n_head,
        'head_size': model.config.n_embd // model.config.n_head,
        'hidden_size': model.config.n_embd,
        'layer_norm_eps': model.config.layer_norm_epsilon,
        'max_position_embeddings': model.config.n_positions,
        'gelu_mode': 'erf' if model.config.activation_function == 'gelu_new' else 'tanh',
    }
    consts = {
        'transformer.wte.weight': model.transformer.wte.weight.detach().numpy(),
        'transformer.wpe.weight': model.transformer.wpe.weight.detach().numpy(),
        'transformer.ln_f.bias': model.transformer.ln_f.bias.detach().numpy(),
        'transformer.ln_f.weight': model.transformer.ln_f.weight.detach().numpy(),
        'layers': [
            {
                'transformer.h.ln_1.bias': l.ln_1.bias.detach().numpy(),
                'transformer.h.ln_1.weight': l.ln_1.weight.detach().numpy(),
                'transformer.h.attn.c_attn.bias': None if l.attn.c_attn.bias is None else l.attn.c_attn.bias.detach().numpy(),
                'transformer.h.attn.c_attn.weight': l.attn.c_attn.weight.transpose(0, 1).contiguous().detach().numpy(),
                'transformer.h.attn.c_proj.bias': None if l.attn.c_proj.bias is None else l.attn.c_proj.bias.detach().numpy(),
                'transformer.h.attn.c_proj.weight': l.attn.c_proj.weight.transpose(0, 1).contiguous().detach().numpy(),
                'transformer.h.ln_2.bias': l.ln_2.bias.detach().numpy(),
                'transformer.h.ln_2.weight': l.ln_2.weight.detach().numpy(),
                'transformer.h.mlp.c_fc.bias': l.mlp.c_fc.bias.detach().numpy(),
                'transformer.h.mlp.c_fc.weight': l.mlp.c_fc.weight.transpose(0, 1).contiguous().detach().numpy(),
                'transformer.h.mlp.c_proj.bias': l.mlp.c_proj.bias.detach().numpy(),
                'transformer.h.mlp.c_proj.weight': l.mlp.c_proj.weight.transpose(0, 1).contiguous().detach().numpy()
            } for l in model.transformer.h
        ],
    }
    if is_cls:
        consts['score.weight'] = model.score.weight.detach().numpy()
        consts['score.bias'] = model.score.bias.detach().numpy() if model.score.bias else None
    else:
        consts['lm_head.weight'] = model.lm_head.weight.detach().numpy()
        consts['lm_head.bias'] = model.lm_head.bias.detach().numpy() if model.lm_head.bias else None

    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='openai-community/gpt2-medium')
    parser.add_argument('--ov_model_path', type=str, nargs='?', default='./models/gpt2-ov')
    parser.add_argument('--compressed_weight', type=bool, nargs='?', default=False)
    parser.add_argument('--quant_type', type=str, nargs='?', default='')
    parser.add_argument('--cls', type=bool, nargs='?', default=False)
    args = parser.parse_args()
    # for compatible, will remove
    if args.compressed_weight:
        print(f'warning: please use "--quant=nncf_w8" instead.')
        if args.quant_type:
            raise ValueError('compressed_weight and quant_type can not be set at the same time.')
        args.quant_type = 'nncf_w8'
    make_configs['quant_type'] = args.quant_type

    if args.quant_type:
        args.ov_model_path = os.path.join(args.ov_model_path, args.quant_type)
    os.makedirs(args.ov_model_path, exist_ok=True)

    configs, consts, pt_model = get_params_from_model(args.org_model_path, args.cls)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    beg = time.time()
    serialize(model, f'{args.ov_model_path}/{OV_XML_FILE_NAME}')
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
    print(f'save tokenzier to "{args.ov_model_path}" ...')
    pt_model.config.pad_token_id = pt_model.config.eos_token_id
    pt_model.config.use_cache = False
    pt_model.config.save_pretrained(args.ov_model_path)
    save_tokenzier(args.org_model_path, args.ov_model_path)
