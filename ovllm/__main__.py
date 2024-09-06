import argparse
import time
import sys
import csv
from pathlib import Path
from contextlib import nullcontext

import multiprocessing
import numa

class WorkSync:
    def __init__(self, total_workers):
        self.barrier = multiprocessing.Barrier(total_workers)
        self.manager = multiprocessing.Manager()
        self.fps = self.manager.dict()
        
    def sync_fps(self, numa_node, fps, is_master):
        self.fps[numa_node] = fps
        self.barrier.wait()
        if is_master:
            fps_details = ""
            total_fps = 0
            for k in wsync.fps:
                fps = wsync.fps[k]
                fps_details += f"{fps:.1f} (NUMA-{k})"
                total_fps += fps
            print(f"==================== [{fps_details}]  Total FPS (2nd token thoughput): {total_fps:.1f}")
        self.barrier.wait()
        

def main(args, wsync = None, numa_node = None, is_master = True):

    title_tag = ""
    if numa_node is not None:
        numa.schedule.run_on_nodes(numa_node)
        numa.memory.set_membind_nodes(numa_node)
        title_tag = f"NUMA #{numa_node}"

    from .greedy_search import OVLLMGreedy
    from .beam_search import OVLLMBeamSearch

    if args.viztracer:
        from viztracer import VizTracer

    with VizTracer(output_file="ovllm_viztracer.json") if args.viztracer else nullcontext() as viztracer:
        ext_path = None
        if sys.platform == 'win32':
            ext_path = ".\\custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll"
        elif sys.platform == 'linux':
            ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
        else:
            print(f"Sample code not supported on platform: {sys.platform}")
            exit(1)

        if args.beam_size == 0:
            ovllm = OVLLMGreedy(args.model, args.bf16, args.hyper_threading, title_tag)
        else:
            ovllm = OVLLMBeamSearch(args.model, args.bf16, args.hyper_threading, title_tag)

        if not args.prompt and not args.prompt_length:
            # nothing specified, will do a smoke test
            args.prompt = ["Oxygen is a", "I'm", "Who is", "Hello,"]

        prompts = {}
        # with open("prompts.json") as f:
        #     prompts = json.load(f)

        print("Start test ...")
        def run_ntimes(args, text, enforce_input_tokens = None, streamer = None):
            results = []
            for round in range(args.repeat):
                result = ovllm.generate(text, beam_size=args.beam_size, new_token_length=args.answer_length, enforce_input_tokens=enforce_input_tokens, streamer = streamer)
                if wsync:
                    wsync.sync_fps(numa_node, result['tok_tput_2nd'], is_master)
                results.append(result)
            return results
        benchmark_data = []

        if args.prompt:
            if args.repeat == 1 and len(args.prompt) == 1:
                # TextStreamer only supports batch size 1
                streamer = None # TextStreamer(ovllm.tokenizer)
            else:
                streamer = None
            # prompt from command line
            text = args.prompt
            print(f'testing beam-size={args.beam_size} prompt="{text[:16]}..."')
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
                    print(f'testing beam-size={args.beam_size}, batch={batch}, length={plen}...')
                    result = run_ntimes(args, ["Hi"] * batch, enforce_input_tokens=plen)

                benchmark_data += result

    if args.output_results:
        with open(args.output_results, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=benchmark_data[0].keys())
            writer.writeheader()
            for data in benchmark_data:
                writer.writerow(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path to model directory, which contains OpenVINO model and tokenzier")
    parser.add_argument('-pl', '--prompt-length', type=str, nargs='+', required=False,
                        help="prompt length: batchxlength or length")
    parser.add_argument('-p', '--prompt', type=str, nargs='+', required=False,
                        help="prompt")
    parser.add_argument('-al', '--answer-length', type=int,
                        default=32, help="generated token length")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("-bf16", "--bf16", action="store_true")
    parser.add_argument("-bs", "--beam-size", type=int, default=0)
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument("-ht", "--hyper-threading", action="store_true")
    parser.add_argument("--output-results", type=str, help="Output results to CSV file")
    parser.add_argument("-v", "--viztracer", action="store_true")
    parser.add_argument('-numa', '--numa', type=int, nargs='+')
    # Parse the argument
    args = parser.parse_args()

    if args.numa:
        numa_cnt = len(args.numa)
        wsync = WorkSync(numa_cnt)
        worker_process_list = []
        for i in range(1, numa_cnt):
            wp = multiprocessing.Process(target=main, args=(args, wsync, args.numa[i], False))
            worker_process_list.append(wp)

        for wp in worker_process_list:
            wp.start()

        main(args, wsync, args.numa[0], True)

        for wp in worker_process_list:
            wp.join()
    else:
        main(args)
