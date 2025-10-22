docker run --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	sam_is \
	python -m examples.instance_segmentation.sam.train $@

    # python -c "import mobile_sam"  # this is to debug importing of mobile_sam
