MODEL_DIR=data/container_cache/models/animate-x
mkdir -p $MODEL_DIR

if [ ! -f $MODEL_DIR/animate-x_ckpt.pth ]; then
    wget -P $MODEL_DIR https://huggingface.co/Shuaishuai0219/Animate-X/resolve/main/animate-x_ckpt.pth
fi
if [ ! -f $MODEL_DIR/v2-1_512-ema-pruned.ckpt ]; then
    wget -P $MODEL_DIR https://huggingface.co/Shuaishuai0219/Animate-X/resolve/main/v2-1_512-ema-pruned.ckpt
fi
if [ ! -f $MODEL_DIR/open_clip_pytorch_model.bin ]; then
    wget -P $MODEL_DIR https://huggingface.co/Shuaishuai0219/Animate-X/resolve/main/open_clip_pytorch_model.bin
fi

mkdir -p data/container_cache/models/DWPose
if [ ! -f data/container_cache/models/DWPose/dw-ll_ucoco_384.onnx ]; then
    wget -P data/container_cache/models/DWPose https://huggingface.co/Shuaishuai0219/Animate-X/resolve/main/dw-ll_ucoco_384.onnx
fi
if [ ! -f data/container_cache/models/DWPose/yolox_l.onnx ]; then
    wget -P data/container_cache/models/DWPose https://huggingface.co/Shuaishuai0219/Animate-X/resolve/main/yolox_l.onnx
fi
