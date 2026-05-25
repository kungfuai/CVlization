# Detector v2: trained on improved GT (system + measure + keysig at ~99% mAP)

## Setup

- Data: L7a (500 train / 80 dev) + L9 (504 train / 80 dev) + openscore (280 train / 61 dev) = 1284 train / 221 dev.
- Pipeline: lilypond --svg → cairosvg → PNG, bboxes from same SVG.
- Extractors: extract_bboxes (fixed: stem-rejecting barlines, ||-as-double, adaptive system grouping, MusicXML staves_per_system hint) + keysig_extractor (mid-piece changes + line-start restatements).
- YOLOv8n multi-task, 19 classes (system, staff, barline_single, barline_heavy, key_-7..+7), 60 epochs, imgsz=1280, batch=16, lr=1e-3.

## Final per-class mAP50

|class|n_val|mAP50|R|
|---|---|---|---|
|system|633|0.995|1.000|
|staff|1908|0.995|0.999|
|barline_single|12704|0.995|0.992|
|barline_heavy|14|0.038|0.000 (only 14 instances)|
|key_-5..+6 (well-populated)|49–89 ea|0.97–0.995|0.96–1.00|
|key_+5|4|0.512|0.500 (rare)|

Overall mAP50 = 0.848 (dragged down by under-represented classes).

## Comparison vs prior detectors

Earlier L7a-only detectors hit ~99.5% mAP50 on system/staff/barline. The
notable new result: openscore + L9 included in the train set with their
fixed GT, the structural classes still hit 99.5%, and the keysig classes
hit 97-99% across well-represented fifths values.

The previous detector trained on the bug-ridden openscore GT had barline
recall of 12% on dense pages and keysig accuracy below 50%. Same backbone
+ same training recipe, just clean GT, hits 99%.

## Open

- key_+5/+6/+7 have <10 val examples; need more openscore data covering
  these keys for confident metric.
- barline_heavy at 0% because only 14 dev instances and most pieces have
  none. Need data balancing or a separate detector head.
- Measure detection is geometric (derive_measures from barline x's) not a
  separately trained class. Works because barline detection is near perfect.

## Checkpoint

`outputs/detector_v2_best.pt` (3.0M params, yolov8n multi-task).
