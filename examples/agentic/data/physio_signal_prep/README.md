# Physio Signal Prep Agent

Preprocess raw physiological waveforms (Sleep-EDF) using a natural-language spec that documents data lineage and cleaning steps, then export Chronos/Moirai-ready time-series artifacts.

## Layout
```
examples/agentic/data/physio_signal_prep/
├── Dockerfile
├── build.sh
├── ingest.sh            # downloads Sleep-EDF samples + manifest
├── preprocess.sh        # TODO: spec-driven cleaning pipeline
├── export.sh            # TODO: Chronos/Moirai exporters
├── requirements.txt
├── scripts/
│   └── build_sleepedf_cache.py
├── specs/               # natural-language spec templates
├── data/raw/            # user-provided sensor dumps
└── outputs/
```

## Quickstart
```bash
cd examples/agentic/data/physio_signal_prep
./build.sh
./ingest.sh --records SC4001E0-PSG.edf SC4002E0-PSG.edf
# Fill out specs/data_spec.sample.md (YAML front matter + notes)
./preprocess.sh specs/data_spec.sample.md
```

This pulls a small subset of Sleep-EDF Expanded data into the centralized cache (`~/.cache/cvlization/data/sleep-edf`) and writes `outputs/sleep_edf_manifest.json` with the train/val split.
`preprocess.sh` parses the YAML front matter in the provided spec, loads cached signals via the `SleepEDFBuilder`, computes basic stats/clip fractions, and writes `outputs/preprocess_summary.json`.
