
source ../../../python_env/bin/activate
source ../../../ov2/openvino/build/install/setupvars.sh

whisper_pt_model=../../../bugs/whisper/pytorchmodel/whisper_pymodel/

python3 ./whisper.py --org_model_path $whisper_pt_model --ov_model_path ./gen/openai/whisper-tiny/ --quant_type f16