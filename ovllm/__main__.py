import argparse
import json
import time
import hashlib
import numpy as np
import sys
import csv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from .llm import OVLLM

def main():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path to model directory, which contains OpenVINO model and tokenzier")
    parser.add_argument('-pl', '--prompt-length', type=str, nargs='+', default=32, required=False,
                        help="prompt length: batchxlength or length")
    parser.add_argument('-p', '--prompt', type=str, nargs='+', required=False,
                        help="prompt")
    parser.add_argument('-al', '--answer-length', type=int,
                        default=32, help="generated token length")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("-bs", "--beam-size", type=int, default=4)
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument("-ht", "--hyper-threading", action="store_true")
    parser.add_argument("--output-results", type=str, help="Output results to CSV file")
    # Parse the argument
    args = parser.parse_args()

    ext_path = None
    if sys.platform == 'win32':
        ext_path = ".\\custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll"
    elif sys.platform == 'linux':
        ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
    else:
        print(f"Sample code not supported on platform: {sys.platform}")
        exit(1)

    ovllm = OVLLM(args.model, args.beam_size, args.bf16, args.hyper_threading)

    prompts = {}
    # with open("prompts.json") as f:
    #     prompts = json.load(f)

    print("Start test ...")
    def run_ntimes(args, text, enforce_input_tokens = None, streamer = None):
        results = []
        for round in range(args.repeat):
            result = ovllm.generate(text, new_token_length=args.answer_length, enforce_input_tokens=enforce_input_tokens, streamer = streamer)
            results.append(result)
        return results
    benchmark_data = []

    if args.prompt:
        if args.repeat == 1 and len(args.prompt) == 1:
            # TextStreamer only supports batch size 1
            streamer = TextStreamer(ovllm.tokenizer)
        else:
            streamer = None
        # prompt from command line
        text = args.prompt
        print(f'testing prompt="{text[:16]}..."')
        result = run_ntimes(args, text, streamer = streamer)
        benchmark_data += result
    else:
        # prompt from json config
        for plen_str in args.prompt_length:
            plen_array = plen_str.split('x')
            plen = int(plen_array[-1])
            batch = 1
            if len(plen_array) > 1:
                batch = int(plen_array[0])
            if str(plen) in prompts:
                text = prompts[str(plen)]
                print(f'testing prompt="{text[:16]}..."')
                result = run_ntimes(args, text * batch)
            else:
                # Prompt with length {plen} is not provided in prompt.json, will forge"
                print(f'testing batch={batch}, length={plen}...')
                result = run_ntimes(args, ["Hi"] * batch, enforce_input_tokens=plen)

            benchmark_data += result

    if args.output_results:
        with open(args.output_results, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=benchmark_data[0].keys())
            writer.writeheader()
            for data in benchmark_data:
                writer.writerow(data)

if __name__ == "__main__":
    main()