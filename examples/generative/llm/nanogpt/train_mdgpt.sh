docker run --gpus=all \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/${CVL_HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
	-v $(pwd)/cvlization:/workspace/cvlization \
	nanogpt \
	python train_mdgpt.py config/train_shakespeare_char.py
