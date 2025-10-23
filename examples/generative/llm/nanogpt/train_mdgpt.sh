docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/cvlization:/workspace/cvlization \
	nanogpt \
	python train_mdgpt.py config/train_shakespeare_char.py
