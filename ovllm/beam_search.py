import numpy as np
from . import utils, utils_cpp
from openvino.runtime import Tensor, Type
from openvino.runtime import opset11 as opset
import time

def topk(array, k, axis=-1):
    sorted=True
    # Use np.argpartition is faster than np.argsort, but do not return the values in order
    # We use array.take because you can specify the axis
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)
    
    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(
            np.argsort(partitioned_scores, axis=axis), axis=axis
        )
        
        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores
    return scores, ind

def process_logits(batch_size, num_beams, logits, beam_scores):
    next_token_logits = logits[:, -1, :]                             # (batch_size*num_beams, 1, vocab_size)
    next_token_scores = utils.logsoftmax(next_token_logits) # (batch_size*num_beams, vocab_size)
    next_token_scores = next_token_scores + beam_scores[:, None]     # (batch_size*num_beams, vocab_size) + (batch_size*num_beams,)
    vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores.reshape(batch_size, num_beams * vocab_size) # (batch_size, num_beams * vocab_size)

    next_token_scores, next_tokens = topk(
        next_token_scores, 2 * num_beams, axis=1
    )
    # topk from `num_beams * vocab_size` possible next-tokens
    #   (batch_size, 2 * num_beams) (batch_size, 2 * num_beams)
    next_indices = np.floor(next_tokens / vocab_size).astype("int32")
    next_tokens = next_tokens % vocab_size

    return next_token_scores, next_tokens, next_indices



class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: int = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: np.ndarray, sum_logprobs: float, beam_indices: np.ndarray):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                highest_attainable_score = best_sum_logprobs / self.max_length**self.length_penalty
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret


class BeamSearch():
    def __init__(self, batch_size: int, num_beams: int, length_penalty: np.float32 = 1.0, max_length: np.int64 = None):
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.num_beam_groups = 1
        self.num_beam_hyps_to_keep = 1
        self.group_size = num_beams # don't support beam group
        self.do_early_stopping = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        self._done = np.array(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=bool
        )

    def process(
        self,
        input_ids: np.ndarray,
        next_scores: np.ndarray,
        next_tokens: np.ndarray,
        next_indices: np.ndarray
    ):
        cur_len = input_ids.shape[-1] + 1
        group_size = self.num_beams # assume num_beam_groups = 1
        next_beam_scores = np.zeros((self.batch_size, group_size), dtype=next_scores.dtype)
        next_beam_tokens = np.zeros((self.batch_size, group_size), dtype=next_tokens.dtype)
        next_beam_indices = np.zeros((self.batch_size, group_size), dtype=next_indices.dtype)

        for batch_idx in range(self.batch_size):
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * group_size + next_index
                next_beam_scores[batch_idx, beam_idx] = next_score
                next_beam_tokens[batch_idx, beam_idx] = next_token
                next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                beam_idx += 1
                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == group_size:
                    break
                    # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )
        return  next_beam_scores.reshape(-1), next_beam_tokens.reshape(-1), next_beam_indices.reshape(-1)
        
    def finalize(
        self,
        input_ids: np.ndarray,
        final_beam_scores: np.ndarray,
        final_beam_tokens: np.ndarray,
        final_beam_indices: np.ndarray,
        max_length: int,
        pad_token_id: int = None,
        eos_token_id: int = None,
        beam_indices: int = None,
    ) -> np.ndarray:
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = np.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []
        best_indices = []
        best_scores = np.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=np.float32)

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded = np.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices = np.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = best_idx

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }

def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array([next_tokens]).reshape(-1, 1)

    if 'attn_mask' in model_inputs:
        attention_mask = model_inputs['attn_mask']
        model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                    np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
    return model_inputs

def change_model_for_beam(model, beam_size):
    
    # print(type(model), dir(model))

    result = model.get_result()
    logits = result.input_value(0)

    # logits : (batch*beam_size, 1, vocab_size)

    #print(type(logits), dir(logits))
    #print(logits.get_partial_shape())
    #raise 1

    topk_beams = beam_size * 2 # opset.parameter([], Type.i32, name='topk_beams')

    oshape = opset.parameter([2], Type.i32, name='oshape')
    oshape.output(0).set_names(set(['oshape']))

    beam_scores = opset.parameter([-1], Type.f32, name='beam_scores')
    beam_scores.output(0).set_names(set(['beam_scores']))
    beam_scores_2d = opset.unsqueeze(beam_scores, np.int32([1]))

    # post-processing for beam-search
    logits_2d = opset.squeeze(logits, 1)                # (batch*beam_size, 1, vocab_size)
    log_softmax = opset.log_softmax(logits_2d, axis=1)
    h_lgsoftmax0 = opset.add(log_softmax, beam_scores_2d)

    #vocab_size = consts['lm_head.weight'].shape[0]
    #nbvoc = opset.multiply(num_beams, np.int32(vocab_size))

    h_lgsoftmax = opset.reshape(h_lgsoftmax0, oshape, special_zero=True)
    h_topk = opset.topk(h_lgsoftmax, topk_beams, axis=np.int32(-1), mode="max", sort="value")
    next_score = opset.result(h_topk.output(0), name='next_score')
    next_indicies = opset.result(h_topk.output(1), name='next_indicies')

    model.add_parameters([oshape, beam_scores])
    model.add_results([next_score, next_indicies])

    return model

kv_cache = None

def generate_beam(model, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id, max_kv_len = 2048, beam_size = 4):
    """
    text prediction cycle.

    Parameters:
      input_ids: tokenized input ids for model
      attention_mask: attention mask for model
      max_sequence_length: maximum sequence length for stop iteration
      eos_token_ids: end of sequence index from vocab
      dynamic_shapes: use dynamic shapes for inference or pad model input to max_sequece_length
    Returns:
      predicted token ids sequence
    """
    import os
    BEAM_DEBUG = int(os.environ.get('BEAM_DEBUG', '0'))
    loop_id = 0

    def debug_log(*args, **kwargs):
        nonlocal loop_id
        if loop_id < BEAM_DEBUG:
            print(f" =======[BEAM_DEBUG:{loop_id}]======= ", end="")
            print(*args, **kwargs)
        return

    debug_log(model.inputs)
    debug_log(model.outputs)

    model_inputs = {}
    batch_size = input_ids.shape[0]
    cur_len = prompt_length = input_ids.shape[1]
    num_beams = beam_size
    first_iteration = True
    kvcache_shape = [2 * model.pipeline_config.n_layers,
                     max_kv_len,
                     batch_size * num_beams,
                     model.pipeline_config.n_head,
                     model.pipeline_config.head_size]
    global kv_cache
    if not kv_cache or (list(kv_cache.shape) != kvcache_shape):
        kv_cache = Tensor(model.input("kv_cache").get_element_type(), kvcache_shape)
    global_beam_idx = np.zeros([batch_size * num_beams, max_kv_len]).astype("int32")
    beam_table = np.zeros([batch_size * num_beams, max_kv_len]).astype("int32")
    sin_tab, cos_tab = utils.create_sinusoidal_positions(max_kv_len, model.pipeline_config.rotary_dims)

    org_input_ids = input_ids
    input_ids = np.repeat(input_ids, num_beams, axis=0)

    original_attention_mask = attention_mask
    attention_mask = np.repeat(attention_mask, num_beams, axis=0)

    model_inputs = {"input_ids": input_ids,
                    "attn_mask": original_attention_mask,
                    "kv_cache": kv_cache,
                    "beam_table": beam_table,
                    "cos_tab": cos_tab,
                    "sin_tab": sin_tab,
                    "oshape" : np.array([batch_size, -1]),
                    "beam_scores" : None,
                    }

    beam_searcher = BeamSearch(batch_size, num_beams)

    logits_output = next(iter(model.outputs))
    vocab_size = logits_output.partial_shape[-1].get_length()

    latency = []
    while True:
        loop_id += 1
        
        time0 = time.time()
        cur_input_len = len(input_ids[0])

        if first_iteration:
            model_inputs['input_ids'] = org_input_ids
            model_inputs['attn_mask'] = original_attention_mask
            model_inputs['beam_scores'] = np.full((batch_size, ), 0, dtype=np.float32) # first iteration has only 1 beam per-batch

            outputs = model(model_inputs)
            logits, next_score, next_indices = outputs.values()      # unpack outputs

            # pretend that we have (batch_size * num_beams) results
            model_inputs["attn_mask"] = np.repeat(original_attention_mask, num_beams, axis=0)
            next_token_logits = np.repeat(logits, num_beams, axis=0)
            beam_scores = np.full((batch_size*num_beams, ), -1e9, dtype=np.float32)
            beam_scores[0::num_beams] = 0
        else:
            outputs = model(model_inputs)
            next_token_logits, next_score, next_indices = outputs.values()      # unpack outputs

        next_tokens = next_indices % vocab_size
        next_indices = np.floor(next_indices / vocab_size).astype("int32")

        debug_log(f"next_token_logits\n", next_token_logits)
        # pre-process distribution
        # next_score, next_tokens, next_indices = process_logits(batch_size, num_beams, next_token_logits, beam_scores)

        debug_log(f"next_score\n", next_score)
        debug_log(f"next_tokens\n", next_tokens)
        debug_log(f"next_indices\n", next_indices)

        beam_scores, beam_next_tokens, beam_idx = beam_searcher.process(input_ids, next_score, next_tokens, next_indices)
        debug_log(f"beam_scores\n", beam_scores)
        debug_log(f"beam_next_tokens\n", beam_next_tokens)
        debug_log(f"beam_idx\n", beam_idx)

        if first_iteration:
            for i in range(cur_len):
                global_beam_idx[:, i] = beam_idx
        global_beam_idx[:, cur_len] = beam_idx
        input_ids = np.concatenate([input_ids[beam_idx, :], np.expand_dims(beam_next_tokens, -1)], axis=-1)

        debug_log(f"beam_idx  :", beam_idx)
        debug_log(f"cur_len+1 :", cur_len+1)

        utils_cpp.update_beam_table(global_beam_idx, beam_table, cur_len+1)
        cur_len = cur_len + 1

        debug_log(f"global_beam_idx \n", global_beam_idx[:,:])
        debug_log(f"beam_table 222222\n", beam_table[:,:])

        #print(f" input_ids.shape={input_ids.shape} beam_idx={beam_idx} kv_cache={kv_cache.shape}")
        # kvcache_shape = [2 * n_layers, batch_size * num_beams, n_head, max_kv_len, head_size]
        debug_log(f"kv_cache = \n",  kv_cache.data[0, 0:8, :,0, 0])

        # model_inputs = prepare_next_input(model_inputs, beam_next_tokens)
        # def prepare_next_input(model_inputs, next_tokens):

        # prepare_next_input
        model_inputs['input_ids'] = np.array([beam_next_tokens]).reshape(-1, 1)

        debug_log("input_ids for next\n", model_inputs['input_ids'])

        if 'attn_mask' in model_inputs:
            attention_mask = model_inputs['attn_mask']
            model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                        np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
        if "beam_scores" in model_inputs:
            model_inputs["beam_scores"] = beam_scores

        if first_iteration:
            debug_log("<===================================================================================== first_iteration is done!!!!!!!!!!!!!!!")
        first_iteration = False

        latency.append(time.time() - time0)
        if cur_len == max_new_tokens + prompt_length:
            break

    sequence_outputs = beam_searcher.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        max_new_tokens + prompt_length,
        pad_token_id,
        eos_token_id
    )
    return sequence_outputs["sequences"], latency

if __name__ == "__main__":
    _beam_search = BeamSearch(1, 4)
    import transformers
    import torch
    pt_beam_search = transformers.generation.BeamSearchScorer(
        1, 4, torch.device("cpu"))

    input_ids = np.ones([4, 12])
    next_token_scores = np.array([-1.2431, -2.2431, -2.8681, -3.1181, -3.2431, -3.2431, -3.7431, -3.8681], dtype=np.float32).reshape([1, 8])
    next_tokens = np.array([198,  1375,  1406,   887, 15640,  2080,  8975,   317], dtype=np.int64).reshape([1, 8])
    next_indices = np.zeros([1, 8], dtype=np.int64)
    pt_output = pt_beam_search.process(torch.from_numpy(input_ids), torch.from_numpy(
        next_token_scores), torch.from_numpy(next_tokens), torch.from_numpy(next_indices))
    _output = _beam_search.process(input_ids, next_token_scores, next_tokens, next_indices)
        
        


