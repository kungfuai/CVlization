import argparse
from pathlib import Path

import cv2
import requests
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_INPUT = "artifacts/sample_input.jpg"
DEFAULT_OUTPUT = "artifacts/panoptic_output.jpg"
SAMPLE_URL = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/demo/input.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detectron2 PanopticFPN inference")
    parser.add_argument("--input", default=None, help="Input image path (default: bundled sample)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold")
    return parser.parse_args()


def ensure_sample_image(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(SAMPLE_URL, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)


def main() -> None:
    args = parse_args()

    if args.input is None:
        input_path = Path(DEFAULT_INPUT)
        ensure_sample_image(input_path)
        print(f"No --input provided, using bundled sample: {input_path}")
    else:
        input_path = Path(resolve_input_path(args.input))

    output_path = Path(resolve_output_path(args.output))

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh

    predictor = DefaultPredictor(cfg)

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")

    outputs = predictor(image)
    panoptic_seg, segments_info = outputs["panoptic_seg"]

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    vis = Visualizer(image[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    vis_output = vis.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_output.get_image()[:, :, ::-1])

    print(f"Panoptic output written to {output_path}")


if __name__ == "__main__":
    main()
