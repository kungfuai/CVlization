# download to data/container_cache/models/wan
model_dir=data/container_cache/models/wan
mkdir -p $model_dir

if [ ! -f $model_dir/umt5_xxl_fp8_e4m3fn_scaled.safetensors ]; then
    wget -P $model_dir https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
fi

if [ ! -f $model_dir/wan_2.1_vae.safetensors ]; then
    wget -P $model_dir https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
fi

if [ ! -f $model_dir/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors ]; then
    wget -P $model_dir https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors
fi

if [ ! -f $model_dir/clip_vision_h.safetensors ]; then
    wget -P $model_dir https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
fi

