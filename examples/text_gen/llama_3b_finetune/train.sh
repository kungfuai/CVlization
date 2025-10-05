docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/llama_3b_finetune:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e HF_TOKEN=$HF_TOKEN \
	llama_3b_finetune \
	python3 train.py
