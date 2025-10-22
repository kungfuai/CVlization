# Download models

ROOT_DIR=examples/video_gen/mimic_motion

## DWPose
mkdir -p $ROOT_DIR/models/DWPose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O $ROOT_DIR/models/DWPose/yolox_l.onnx
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O $ROOT_DIR/models/DWPose/dw-ll_ucoco_384.onnx

## MimicMotion
wget -P $ROOT_DIR/models/ https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth