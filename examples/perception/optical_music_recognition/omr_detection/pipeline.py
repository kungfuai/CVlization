"""End-to-end inference orchestrator for the detection-first pipeline.

Flow (planned):
  1. Load layout detector and key-sig classifier.
  2. For each input page image:
       a. detector.predict(image)
            -> systems, staves, barlines, [key_sigs, clefs]
       b. derive cells = staff_bbox ∩ (between consecutive barlines)
       c. for each cell:
            - crop the cell region from the page
            - per-cell transcription model (TODO) -> MXC2 of that cell
       d. determine per-cell key:
            - if key_sig was detected for that staff's system, use it
            - else use the most-recent prior key_sig (key carries forward)
            - run keysig_cnn on the cropped key-sig region as fallback
       e. apply respell.py per cell using the per-cell key
  3. Stitch per-cell MXC2 back into page-level MXC2 using bar numbers
     and part identity (top staff = part 1, etc.).

Status: STUB — depends on detector + per-cell transcriber.

For now, the only existing piece is the key-sig classifier. The per-cell
transcriber is the next decision point: lift the safckylj-style VLM but
restrict its input to a cell? Train a smaller seq2seq model on
measure-sized targets? See README "Build order".
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--page-image", required=True)
    p.add_argument("--detector-ckpt", required=True)
    p.add_argument("--keysig-ckpt", required=True)
    p.add_argument("--transcriber-ckpt", default=None,
                   help="per-cell transcription model (TBD)")
    args = p.parse_args()
    raise NotImplementedError(
        "Build detector + per-cell transcriber first, then wire here.")


if __name__ == "__main__":
    main()
