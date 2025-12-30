## Quick Start

```bash
bash examples/generative/video_generation/worldcanvas/build.sh
```

```bash
bash examples/generative/video_generation/worldcanvas/predict.sh --steps 50
```

Outputs are written to `examples/generative/video_generation/worldcanvas/outputs`.

## Notes

- Model assets are lazily downloaded on first run via `huggingface_hub` and cached under `~/.cache/huggingface`.
- Defaults use samples stored at `zzsi/cvl/worldcanvas/` on Hugging Face Datasets.
- Override repositories with `WORLDCANVAS_T5_VAE_REPO`, `WORLDCANVAS_DIT_REPO`, `WORLDCANVAS_TOKENIZER_DATASET_REPO`, and `WORLDCANVAS_TOKENIZER_DATASET_PATH`.

## License

WorldCanvas is licensed under CC BY-NC-SA 4.0 (non-commercial, share-alike). The original repository states the code is for academic research only. Please review the upstream license before use.
