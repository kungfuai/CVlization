docker run --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	dreambooth \
	python -m examples.image_gen.dreambooth.train --save_sample_prompt "a happy sks cat" $@
