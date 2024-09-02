import argparse
import json
import time
import hashlib
import numpy as np
import sys
import csv
from pathlib import Path
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from .greedy_search import generate_greedy, change_model_for_greedy
from .beam_search import generate_beam, change_model_for_beam
from .export.utils import OV_XML_FILE_NAME

class ModelConfig:
    def __init__(self, ov_model) -> None:
        kv_cache_shape = ov_model.input("kv_cache").partial_shape
        cos_tab_shape = ov_model.input("cos_tab").partial_shape

        # 2*n_layers, L, B, H, S
        self.n_layers = kv_cache_shape[0].get_length() // 2
        self.n_head = kv_cache_shape[3].get_length()
        self.head_size = kv_cache_shape[4].get_length()
        self.rotary_dims = cos_tab_shape[1].get_length() # assumes sin/cos table dims is half of rotary_dims

    def __str__(self) -> str:
        return f"\tn_layers={self.n_layers}, n_head={self.n_head}, head_size={self.head_size}, rotary_dims={self.rotary_dims}"

def post_processing(result, input_text): 
    """post processing the model output"""
    ans = result
    if result[:len(input_text)] == input_text:
        ans = result[len(input_text):]
    return ans

class OVLLM(object):
    def __init__(self, model_path, beam_size = 0, use_infer_prec_bf16 = True, hyper_threading = False):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # This is not supported anymore with the latest transformers (https://github.com/huggingface/transformers/pull/23909)
            #tokenizer.pad_token = tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"             # pad to left

        # initialize openvino core
        self.core = Core()
        print("Init OpenVINO model ...")
        # read the model and corresponding weights from file
        self.ov_model = self.core.read_model(Path(model_path) / OV_XML_FILE_NAME)

        # add preprocessor for bf16 kv_cache
        if use_infer_prec_bf16:
            kv_cache_precision = Type.bf16
            ppp = PrePostProcessor(self.ov_model)
            for key in self.ov_model.inputs:
                if "kv_cache" in key.get_any_name() and kv_cache_precision != key.get_element_type():
                    ppp.input(key.get_any_name()).tensor().set_element_type(kv_cache_precision)
            self.ov_model = ppp.build()
        
        ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                    "INFERENCE_PRECISION_HINT" : "bf16" if use_infer_prec_bf16 else "f32",
                    "CPU_DENORMALS_OPTIMIZATION" : "YES",
                    "ENABLE_HYPER_THREADING" : "YES" if hyper_threading else "NO",
                    "CACHE_DIR" : None}

        if beam_size > 0:
            self.ov_model = change_model_for_beam(self.ov_model, beam_size)
        else:
            self.ov_model = change_model_for_greedy(self.ov_model)

        self.compiled_model = self.core.compile_model(self.ov_model, "CPU", ov_config)
        self.compiled_model.pipeline_config = ModelConfig(self.ov_model)
        self.beam_size = beam_size
        self.last_output_text_map = {}

    def generate(self, text, new_token_length, enforce_input_tokens = None, streamer = None, continuation = None):
        """
         enforce_input_tokens : enfore input token length to be of this number
        """
        if enforce_input_tokens:
            # repeat text up-to enforce_input_tokens length 
            inputs = self.tokenizer(text, return_tensors="np", padding=True, return_token_type_ids=False)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            attention_mask = (1.0 - attention_mask) * np.finfo(np.float32).min

            input_ids = np.tile(input_ids[:, 0:1], enforce_input_tokens)
            attention_mask = np.tile(attention_mask[:, 0:1], enforce_input_tokens)

            input_token_len = input_ids.shape[1]
            input_batch_size = input_ids.shape[0]
        else:
            inputs = self.tokenizer(text, return_tensors="np", padding=True, return_token_type_ids=False)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            attention_mask = (1.0 - attention_mask) * np.finfo(np.float32).min

            input_token_len = input_ids.shape[1]
            input_batch_size = input_ids.shape[0]

        if continuation:
            # override new_token_length
            assert(type(text) is list)
            assert(type(continuation.text) is list)
            assert(len(continuation.text) == len(text))
            assert(self.beam_size == 0)

            new_token_length = 0
            for t, c in zip(text, continuation.text):
                t_toks = self.tokenizer(t, return_tensors="np", padding=True, return_token_type_ids=False)["input_ids"]
                tc_toks = self.tokenizer(t + c, return_tensors="np", padding=True, return_token_type_ids=False)["input_ids"]
                continuation.tokens.append(tc_toks[0, t_toks.shape[1]:])
                new_token_length = max(new_token_length, tc_toks.shape[1] - t_toks.shape[1])

        gen_sequence_start = time.time()
        if self.beam_size == 0:
            output_ids, latency = generate_greedy(self.compiled_model, input_ids, attention_mask, 
                                        max_new_tokens=new_token_length,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        max_kv_len=input_token_len + new_token_length*2,
                                        streamer = streamer,
                                        continuation = continuation)
        else:
            output_ids, latency = generate_beam(self.compiled_model, input_ids, attention_mask, 
                                        max_new_tokens=new_token_length,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        max_kv_len=input_token_len + new_token_length*2,
                                        beam_size=self.beam_size)

        if continuation:
            return

        gen_sequence_end = time.time()
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        gen_sequence_length = len(output_ids[0]) - len(input_ids[0])
        gen_latency = gen_sequence_end - gen_sequence_start

        n_latency = len(latency)
        token_total = sum(latency)

        average_token_latency = sum(latency[2:])/(n_latency-2)
        overhead_latency = gen_latency - token_total
        
        print(f"  [{input_batch_size}x{self.beam_size}, {input_token_len:4}+{gen_sequence_length}]  {gen_latency*1e3:.1f}ms = {latency[0]*1e3:.1f}ms + {latency[1]*1e3:.1f}ms + ({average_token_latency*1e3:.1f}ms x {n_latency-2}) + {overhead_latency * 1e3:.1f}ms")

        text_key = ",".join(text)

        if text_key not in self.last_output_text_map or self.last_output_text_map[text_key] != output_text:
            self.last_output_text_map[text_key] = output_text
            for i, out in enumerate(output_text):
                md5sum = hashlib.md5(out.encode('utf-8')).hexdigest()
                console_out = post_processing(out, text)
                if len(console_out) > 160:
                    console_out = console_out[:80] + "..." + md5sum
                print(f"\t{i}. {[console_out]}")

        benchmark_data = {
            'input_batch_size': input_batch_size,
            'input_token_length': input_token_len,
            'generated_sequence_length': gen_sequence_length,
            'generation_latency_total_ms': gen_latency * 1e3,
            'token_latency_first_ms': latency[0] * 1e3,
            'average_token_latency_ms': average_token_latency * 1e3,
            'overhead_ms': overhead_latency * 1e3,
            'output': post_processing(output_text[0], text)
        }

        return benchmark_data