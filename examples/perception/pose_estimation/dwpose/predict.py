import os
from PIL import Image
import numpy as np
from dwpose_lib.pose import get_image_pose, get_video_pose


def download_weights(cache_dir: str = "/root/.cache"):
    """
    Download the weights for the model.

    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "models/DWPose"), exist_ok=True)
    # Check if the files already exist
    if not os.path.exists(os.path.join(cache_dir, "models/DWPose/yolox_l.onnx")):
        print("Downloading yolox_l.onnx")
        os.system(f"wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O {os.path.join(cache_dir, 'models/DWPose/yolox_l.onnx')}")
    if not os.path.exists(os.path.join(cache_dir, "models/DWPose/dw-ll_ucoco_384.onnx")):
        print("Downloading dw-ll_ucoco_384.onnx")
        os.system(f"wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O {os.path.join(cache_dir, 'models/DWPose/dw-ll_ucoco_384.onnx')}")


def main():
    download_weights()
    ## Test on a video
    video_path = r"examples/pose1.mp4"
    ref_pose, pose_img = get_video_pose(video_path, draw=False)
    def shape_or_value(x):
        if hasattr(x, "shape"):
            return x.shape
        else:
            return x
    print("outputs: pose:", {k: shape_or_value(v) for k, v in ref_pose.items()}, "pose_img:", pose_img.shape if pose_img is not None else None)
    print("example values for body pose:", ref_pose["body"][0, :5, :])
    print("example values for face pose:", ref_pose["face"][0, :5, :])

    ## Test on an image
    if False:
        ref_image = Image.open(r"examples/human.png").convert("RGB")
        ref_image = np.array(ref_image)
        print("input shape:", ref_image.shape)
        ref_pose, pose_img = get_image_pose(ref_image)
        print("outputs: pose:", ref_pose, "pose_img:", pose_img.shape)
        print("bodies:", ref_pose["bodies"].keys())
        print("bodies candidates:", ref_pose["bodies"]["candidate"].shape)
        print("body subsets:", ref_pose["bodies"]["subset"].shape)
        print(ref_pose["bodies"]["subset"])
        print("body scores:", ref_pose["bodies"]["score"].shape)
        print("faces:", ref_pose["faces"].shape)
        # print("hands:", ref_pose["hands"].keys())
        print("pose keys:", ref_pose.keys())


if __name__ == "__main__":
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    print("providers:", providers)

    main()
