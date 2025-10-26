docker run --gpus=all \
	-v $(pwd)/examples/text_gen/mixtral8x7b/generate.py:/workspace/generate.py \
	-v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
	-e HF_TOKEN=$HF_TOKEN \
	mixtral \
	python3 generate.py
