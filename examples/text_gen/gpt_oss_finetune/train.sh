docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/gpt_oss_finetune:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e HF_TOKEN=$HF_TOKEN \
	gpt_oss_finetune \
	python3 train.py
