#!/usr/bin/env python3
"""Train the layout detector.

Usage (planned):
    python train_detector.py --config configs/detector_l7a.yaml

Status: STUB — wire up once a backbone is chosen and the detection
dataset is built (`labels/make_dataset.py`).
"""
import argparse
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)
    raise NotImplementedError("Pick backbone + implement training loop.")


if __name__ == "__main__":
    main()
