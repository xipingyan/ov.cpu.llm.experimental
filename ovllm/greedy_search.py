import numpy as np
from . import utils
import time
from openvino.runtime import Tensor, Type
from openvino.runtime import opset11 as opset
from . import llm

class OVLLMGreedy(llm.OVLLM):

    def _patch_model(self, model):
        result = model.get_result()
        logits = result.input_value(0)
        # logits : (batch*beam_size, 1, vocab_size)

        topk_K = opset.parameter([], Type.i32, name='topk_K')
        topk_K.output(0).set_names(set(['topk_K']))

        h_topk = opset.topk(logits, topk_K, axis=np.int32(-1), mode="max", sort="value")
        next_score = opset.result(h_topk.output(0), name='next_score')
        next_indicies = opset.result(h_topk.output(1), name='next_indicies')

        model.add_parameters([topk_K])
        model.add_results([next_score, next_indicies])
        return model

    def _generate(self,
                  model,    # ompiled_model,
                  kv_cache, # [2 * n_layers, max_kv_len, batch_size * beam_size, n_head, head_size]
                  eos_token_id,
                  pad_token_id,
                  input_ids,
                  attention_mask,
                  beam_size,
                  max_new_tokens,
                  max_kv_len,
                  streamer,
                  continuation):
        batch_size = input_ids.shape[0]

        # initialize "straight" beams in greedy search
        beam_table = np.zeros([0, max_kv_len]).astype("int32")

        sin_tab, cos_tab = utils.create_sinusoidal_positions(max_kv_len, self.pipeline_config.rotary_dims, self._get_rotary_base())
        model_inputs = {"input_ids": input_ids,
                        "attn_mask": attention_mask,
                        "kv_cache": kv_cache,
                        "beam_table": beam_table,
                        "cos_tab": cos_tab,
                        "sin_tab": sin_tab,
                        "topk_K" : np.int32(1),
                        }
        latency = []
        cur_len = 0

        if streamer:
            print("\033[0;32m")
            streamer.put(input_ids)
            streamer.end()
            print("\033[0;33m")

        while True:
            time0 = time.time()
            outputs = model(model_inputs)

            logits, next_score, next_indicies = outputs.values()

            next_token_logits = logits[:, -1, :]
            
            # pre-process distribution
            next_tokens_scores = next_token_logits
            # next_tokens = np.argmax(next_tokens_scores, axis=-1)
            next_tokens = next_indicies[:,0,0]

            if continuation:
                # next token has been determined by continuation_tokens
                # collect logprobs and replace next
                logprobs = utils.logsoftmax(next_token_logits)
                for b in range(logprobs.shape[0]):
                    if cur_len < len(continuation.tokens[b]):
                        expect_tok = continuation.tokens[b][cur_len]
                        continuation.logprobs[b] += logprobs[b, expect_tok]
                        continuation.is_greedy[b] = continuation.is_greedy[b] & (next_tokens[b] == expect_tok)
                        # replace next token with expected one
                        next_tokens[b] = expect_tok
            # get next token id
            # break the loop if max length or end of text token is reached
            cur_len = cur_len + 1
            if cur_len == max_new_tokens or (next_tokens == eos_token_id).all():
                latency.append(time.time() - time0)
                break

            if streamer and len(next_tokens) == 1:
                streamer.put(next_tokens)
            input_ids = np.concatenate((input_ids, next_tokens[:, None]), axis=-1)

            model_inputs['input_ids'] = np.array(next_tokens[..., np.newaxis])
            if 'attn_mask' in model_inputs:
                attention_mask = model_inputs['attn_mask']
                model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                            np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)        

            latency.append(time.time() - time0)

        if streamer:
            streamer.end()
            print("\033[00m")
        return input_ids, latency