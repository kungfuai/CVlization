---
source:
  dataset: sleep-edf expanded
  subset: sleep-cassette
  include_globs:
    - data/raw/sleep-edf/**
    - ~/.cache/cvlization/data/sleep-edf/sleep-cassette/**/*.edf
  exclude_globs:
    - "**/bad_subjects/*"
  record_patterns:
    - "SC40*E0-PSG.edf"
channels:
  required:
    - Fpz-Cz
    - Pz-Oz
  optional:
    - C3-M2
    - C4-M1
  exclude: []
  pad_missing_channels: true
sampling:
  target_hz: 50
  source_hz:
    default: 100
    overrides:
      - pattern: "**/telemetry/**"
        hz: 200
    timestamps:
      - file: data/raw/sleep-edf/SC4001E0-timestamps.csv
normalization:
  strategy: zscore
  scope: per_sequence_per_channel
  reference_split: training
  group_by: ["modality"]
windowing:
  length_seconds: 30
  overlap_seconds: 0
filters:
  bandpass: [0.3, 35.0]
  notch:
    hz: 50
    skip_if_outside_band: true
qc:
  amplitude_clip_uv: 500
  interruption_rate_threshold: 0.1
artifact_handling:
  policy: drop_segments
  notes:
    - discard epochs with obvious movement artifacts or technician marks
export:
  format: chronos
  output_dir: processed
---

# Physio Signal Prep Spec (Sample)

## Narrative Notes
- Source selection: point loader at cached Sleep-EDF cassette folder; include SC40xx adults and drop any subject-specific quality issues listed above.
- Sampling: default 100 Hz; telemetry subset is 200 Hz. If uneven sampling, use the provided timestamp CSV instead of a fixed source Hz.
- Filters: 0.3–35 Hz band; apply 50 Hz notch only when it lies inside the bandpass.
- Windowing: 30 s windows, no overlap (typical for sleep staging).
- Normalization: z-score per sequence per channel using training split stats; regroup by modality if mixing EEG/EOG/EMG.
- Artifacts: discard windows with technician-marked artifacts or extreme amplitude; note any removed intervals here so downstream tasks can ignore them.
