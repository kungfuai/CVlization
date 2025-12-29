import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from cvlization.paths import get_output_dir
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


def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj


def main():
    download_weights()

    # Create output directory
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)

    ## Test on a video
    print("Processing video: examples/pose1.mp4")
    video_path = r"examples/pose1.mp4"
    ref_pose, pose_img = get_video_pose(video_path, draw=False)

    # Print shapes for debugging
    def shape_or_value(x):
        if hasattr(x, "shape"):
            return x.shape
        else:
            return type(x).__name__

    print(f"Video output shapes - pose: {shape_or_value(ref_pose)}, pose_img: {pose_img.shape if pose_img is not None else None}")

    # Print sample values safely
    if isinstance(ref_pose, dict):
        for key in ref_pose:
            val = ref_pose[key]
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], np.ndarray):
                    print(f"  {key}: list of {len(val)} arrays, first shape: {val[0].shape}")
                    if len(val[0]) > 0:
                        print(f"    Sample from first frame: {val[0][:2]}")
            elif isinstance(val, np.ndarray):
                print(f"  {key}: array shape {val.shape}")
                if len(val) > 0:
                    print(f"    Sample: {val[:2]}")

    # Save video pose output
    video_output = output_dir / "pose1_video_output.json"
    with open(video_output, 'w') as f:
        json.dump(numpy_to_list(ref_pose), f, indent=2)
    print(f"\nSaved video pose output to: {video_output}")

    ## Test on an image
    print("\nProcessing image: examples/human.png")
    ref_image = Image.open(r"examples/human.png").convert("RGB")
    ref_image = np.array(ref_image)
    print(f"Input image shape: {ref_image.shape}")

    ref_pose_img, pose_img = get_image_pose(ref_image)
    print(f"Image output - pose type: {type(ref_pose_img).__name__}")

    if isinstance(ref_pose_img, dict):
        print("Pose keys:", ref_pose_img.keys())
        if "bodies" in ref_pose_img:
            print("  bodies keys:", ref_pose_img["bodies"].keys())
            for key in ref_pose_img["bodies"]:
                val = ref_pose_img["bodies"][key]
                if hasattr(val, "shape"):
                    print(f"    {key} shape: {val.shape}")

    # Save image pose output
    image_output = output_dir / "human_image_output.json"
    with open(image_output, 'w') as f:
        json.dump(numpy_to_list(ref_pose_img), f, indent=2)
    print(f"Saved image pose output to: {image_output}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    print("providers:", providers)

    main()
