# ov.cpu.llm.experimental
This repo demonstrates a LLM optimization method by custom-ops for OpenVINO. In order to inference the LLM efficiently, this repo introduces a new Op called `MHA` and re-construct the LLM based on this new-ops.

This environment and benchmark can be built in a Docker environment (section 1), or inside a Linux/Windows bare metal system (section 2).

## 1. Docker-based Environment

### 1.1 Build Docker image

Build a docker image that has all dependencies, custom OpenVINO and custom ops builds installed:

```bash
docker build -t llm-openvino .
```

### 1.2 Generate optimized model with Docker

With this built Docker image, you can then generate the optimized OpenVINO IR (optionally with weight compression) as follows. Note that here we're assuming that you have already downloaded a model from the Huggingface hub into a local cache on the host (e.g. using `hugginface-cli download`), and volume mounting it into the Docker container using the Docker `-v` argument as illustrated below.

```bash
mkdir -p $HOME/models
docker run --rm -v $HOME/.cache/huggingface:/cache/huggingface -v $HOME/models:/models -it openvino-llm \
  python3 models/llama.py \
    --quant_type nncf_w8 \
    --org_model_path /cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235 \
    --ov_model_path /models/llama-2-7b-chat-ov
```

This will put the optimized OpenVINO IR (and associated tokenizer) into `~/models/llama-2-7b-chat-ov` on the host. 

### 1.2 Benchmark optimized model with Docker

Now we're ready to run the benchmarks using this model, again mounting that directory into the launched docker container. The following command runs the benchmark of the compressed model for 3 iterations, with BF16 precision.

```bash
docker run --privileged --rm -v $HOME/models:/models -v $(pwd):/results -it openvino-llm \
  python3 -m ovllm -m /models/llama-2-7b-chat-ov/nncf_w8 --bf16 -r 3 --greedy -p "What is OpenVINO?" --output-results /results/results.csv
```

A sample output is below:
```
[setupvars.sh] OpenVINO environment initialized
Using pad_token, but it is not set yet.
Init OpenVINO model ...
VNode_14200 created executor: llm::experimental::MultiHeadAttention,LLMDNN,BF16
Start test ...
round 0:
  [1,  689+15]  4685.5ms = 3773.2ms + 245.3ms + (47.6ms x 14) + 1.2ms
        0. [' Hello! I am an AI assistant, How can I help you?']
...
```
Since we applied `--output-results` above, you will find the results in the `results.csv` file.

See section 2 below for more examples of model generation and benchmark options.

## 2. Bare Metal Environment

### 2.1. Build Dependency on Linux
You could refer to [build_linux](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md) for more details. Please set the install dir for openvino. Note, please make sure the gcc version is at least 11.2.

#### Build OpenVINO
```bash
git clone https://github.com/usstq/openvino.git -b vnode-lc
cd openvino && git submodule update --init --recursive 
python3 -m pip install -U pip 
python3 -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..

# if want to run the model on multiple numa nodes, use the following
# cmake -DENABLE_INTEL_GPU=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
make --jobs=$(nproc --all)
make install
cd <ov install dir>/tools/
python3 -m pip install  openvino*.whl

```
#### Build Custom Ops Library
Please Do Reminder to enable the customized OpenVINO environment for this repo
```bash
source <ov install dir>/setupvars.sh
cd custom_ops
mkdir build && cd build
cmake ..
make -j8
# custom_ops/build/libov-cpu-llm-experimental.so
```

### 2.2. Build Dependency on Windows
You could refer to [build_windows](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_windows.md) for more details. Please set the install dir for openvino. Note, please make sure the MSVC version is at least Visual Studio 16 2019.

#### Build OpenVINO
```bash
git clone https://github.com/usstq/openvino.git -b vnode-lc
cd openvino && git submodule update --init --recursive
python3 -m pip install -U pip
python3 -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..

# if want to run the model on multiple numa nodes, use the following
# cmake -G "Visual Studio 16 2019" -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
cmake --build . --config Release --verbose -j8
cmake --install .
cd <ov install dir>/tools/
python3 -m pip install  openvino*.whl
```
#### Build Custom Ops Library
Please Do Reminder to enable the customized OpenVINO environment for this repo
```bash
<ov install dir>/setupvars.bat
cd custom_ops
mkdir build && cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release --verbose -j8
# custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll
```

### 2.3. Setup Demo Environment
Install python env
```bash
pip3 install -r requirements.txt
pip3 install -e .
```

### 2.4. Model Conversion
convert orginal model into OpenVINO ovllm IR:

```bash
python -m ovllm.export.gptj --quant_type=Q4_1 # valid types: F16/Q8_C/Q4_C/Q8_0/Q4_0/Q4_1/nncf_w8
python -m ovllm.export.gptneox ...
python -m ovllm.export.falcon
python -m ovllm.export.llama
python -m ovllm.export.chatglm2
```

### 2.5. Run Pipeline

```bash
# greedy search:  f32/bf16 
numactl -N 0 --membind=0  python -m ovllm -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --greedy
numactl -N 0 --membind=0  python -m ovllm -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --greedy --bf16
# beam search:  f32/bf16 
numactl -N 0 --membind=0  python -m ovllm -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3
numactl -N 0 --membind=0  python -m ovllm -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --bf16
# specific input token length (support multiple langth, multiple round)
numactl -N 0 --membind=0  python -m ovllm -m ./gen/gptj_6b/ -pl 32 512 1024 2016 8192 -r 3 --bf16
# run on all numa nodes
python -m ovllm -m ./gen/falcon_40b -bs 1 --bf16 -pl 8000

# test lambada_openai accuracy use lm-evaluation-harness
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-7b-chat/f16/,nbatch=16

```

# Quantization with experimental FC node

Inspired by excellent project [llama.cpp](https://github.com/ggerganov/llama.cpp), we use following quantization methods: 
  - Weights are quantized off-line
  - Activations are quantized dynamically at runtime

| quant_type    |  description |
| ---------     |     -------  |
| `F16`         | FP16 weight format |
| `Q8_C`        | per-output channel symmetric weight-quantization |
| `Q4_C`        | per-output channel asymmetric weight-quantization |
| `Q8_0`, `Q4_0`| llama.cpp style per-32 weights symmetric weight-quantization |
| `Q4_1`        | llama.cpp style per-32 weights asymmetric weight-quantization |
| `nncf_w8`     | per-output channel asymmetric weight-quantization from nncf |

> Note
>  - asymmetric quantization improves accuracy (PPL) at lower quantization bits, so Q4_C uses asymmetric quantization (with integer zero-point which has higher accuracy than non-integer zero-point)

## performance/accuracy report

```bash
# performance
numactl -C0-15  python -m ovllm -m ./gen/llama-2-7b-chat/Q8_0/ -p "I am retail store manager with new ice cream flavor Super Sweet White Coffee. Can you generate a twitter post to promote it?" -r 1 --greedy -al 32

# perplexity
# download wikitext-2-raw from :
#   https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
#   https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
python ./llm_perplexity.py -f=./wikitext-2-raw/wiki.test.raw -ov ./gen/llama-2-7b-chat/F16/
```

