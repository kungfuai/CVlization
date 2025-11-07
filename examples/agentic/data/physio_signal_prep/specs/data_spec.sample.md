---
source:
  dataset: sleep-edf expanded
  subset: sleep-cassette
  record_ids:
    - SC4001E0-PSG.edf
channels:
  include:
    - Fpz-Cz
    - Pz-Oz
sampling:
  original_hz: 100
  target_hz: 50
windowing:
  length_seconds: 30
  overlap_seconds: 15
filters:
  bandpass: [0.3, 35.0]
  notch: 50
normalization: zscore_per_channel
qc:
  amplitude_clip_uv: 500
  interruption_rate_threshold: 0.1
export:
  format: chronos
  output_dir: outputs/processed
---

# Physio Signal Prep Spec (Sample)

## Narrative Notes
- Source data: Sleep-EDF cassette record `SC4001E0`, reference to mastoid.
- Known issues: Eye blinks near lights-off (23:02) and occasional movement artifacts when annotations occur.
- After filtering, resample to 50 Hz, window into 30 s epochs (15 s overlap) and export Chronos-ready parquet/CSV plus metadata YAML.
