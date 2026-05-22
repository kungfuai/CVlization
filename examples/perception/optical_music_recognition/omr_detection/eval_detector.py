#!/usr/bin/env python3
"""Evaluate the layout detector.

Metrics:
  - mAP @ IoU=0.5 per class.
  - Per-class precision/recall.
  - Sanity: cells derivable from staves + barlines must cover the page
    (= measure_count from MusicXML).

Usage (planned):
    python eval_detector.py --checkpoint outputs/detector_l7a/best.pt \\
        --split dev -n 100

Status: STUB.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("-n", "--n-samples", type=int, default=100)
    args = p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
