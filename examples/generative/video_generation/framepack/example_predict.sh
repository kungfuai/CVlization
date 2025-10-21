bash examples/video_gen/framepack/predict.sh \
	--mode i2v --input_image data/1.jpg --prompt "A character doing some simple body movements" --total_seconds 10.0 --seed 42 --steps 30 --output_dir ./data/
	

## Use the following for video extension 
# python predict_f1_enhanced.py \
    # --mode extend \
    # --extension_method f1_multiframe \
    # --input_video "data/1.mp4" \
    # --prompt "The character continues dancing gracefully" \
    # --extend_seconds 3.0 \
	# --output_dir ./data/