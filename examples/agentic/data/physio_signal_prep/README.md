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

### Using cvl CLI (recommended)
```bash
# With Docker (default)
cvl run physio_signal_prep build
cvl run physio_signal_prep ingest --records SC4001E0-PSG.edf SC4002E0-PSG.edf
cvl run physio_signal_prep preprocess

# Without Docker (requires local Python environment with dependencies)
pip install -r examples/agentic/data/physio_signal_prep/requirements.txt
cvl run physio_signal_prep --no-docker ingest --records SC4001E0-PSG.edf
cvl run physio_signal_prep --no-docker preprocess
```

### Using shell scripts directly
```bash
cd examples/agentic/data/physio_signal_prep

# With Docker
./build.sh
./ingest.sh --records SC4001E0-PSG.edf SC4002E0-PSG.edf
./preprocess.sh specs/data_spec.sample.md
# LLM-assisted summary (requires OPENAI_API_KEY)
./preprocess.sh specs/data_spec.sample.md --llm-provider openai --llm-model gpt-5-nano

# Without Docker
pip install -r requirements.txt
CVL_NO_DOCKER=1 ./ingest.sh --records SC4001E0-PSG.edf SC4002E0-PSG.edf
CVL_NO_DOCKER=1 ./preprocess.sh specs/data_spec.sample.md
```

This pulls a small subset of Sleep-EDF Expanded data into the centralized cache (`~/.cache/cvlization/data/sleep-edf`) and writes `outputs/sleep_edf_manifest.json` with the train/val split.
`preprocess.sh` parses the YAML front matter in the provided spec, loads cached signals via the `SleepEDFBuilder`, resamples/window/normalizes them per spec, emits generic time-series parquet files (compatible with Chronos/Moirai or other forecasting pipelines), and writes `outputs/preprocess_summary.json`. When `--llm-provider` is supplied, the script prints which provider/model is used and adds an LLM-generated narrative summary to the output (falls back to spec-only processing when omitted).
