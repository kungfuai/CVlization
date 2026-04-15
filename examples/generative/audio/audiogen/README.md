# AudioGen

Text-to-sound generation using Meta AudioCraft AudioGen.

The default model is `facebook/audiogen-medium`, the released AudioGen model in AudioCraft.

## What to Expect

Running `cvl run audiogen predict` or `./predict.sh` generates a WAV sound effect from a text prompt and saves it in your current working directory. The default output is `audiogen.wav`.

The first real model run downloads `facebook/audiogen-medium` from Hugging Face and caches it under `~/.cache/huggingface`. This is a medium-sized AudioCraft model, so expect a multi-GB first-run download and higher VRAM use than the debug smoke path. The default generation length is 10 seconds so command-line demos produce enough audio to evaluate.

The bundled sample input is a small sirens audio prompt from the upstream AudioCraft repo. It is hosted in `zzsi/cvl` and downloaded lazily only when requested with `--prompt-audio sample`.

## Sample

**Input** — text prompt:

```text
rain falling on a metal roof
```

**Optional audio prompt** — upstream AudioCraft sirens sample:

https://huggingface.co/datasets/zzsi/cvl/resolve/main/audiogen/sample_data/sirens_and_a_humming_engine_approach_and_pass.mp3

**Output** — a WAV file such as `sound.wav` or `audiogen.wav`.

## Usage

Build the Docker image:

```bash
./build.sh
```

Generate a sound effect from text:

```bash
./predict.sh \
  --text "rain falling on a metal roof with distant thunder and water running through gutters" \
  --duration 10 \
  --seed 20260415 \
  --output sound.wav
```

Generate from a text file:

```bash
./predict.sh --input prompt.txt --output sound.wav
```

Generate continuation from the bundled audio prompt:

```bash
./predict.sh \
  --text "distant emergency sirens passing through a city street with engine noise and light traffic" \
  --duration 10 \
  --prompt-audio sample \
  --prompt-duration 2 \
  --output continued_sound.wav
```

Run the smoke test:

```bash
./test.sh
```

The smoke test uses AudioCraft's small `debug` model and verifies bundled sample prompt download plus WAV writing. It does not download `facebook/audiogen-medium`.

## Options

- `--text`: Prompt text.
- `--input`: Path to a text prompt file.
- `--output`: Output WAV path.
- `--model`: Hugging Face model id. Defaults to `facebook/audiogen-medium`.
- `--duration`: Generated duration in seconds. Defaults to `10`.
- `--prompt-audio`: Optional audio prompt for continuation. Use `sample` to download the bundled sirens sample.
- `--prompt-duration`: Seconds to use from the audio prompt. Defaults to `2`.
- `--temperature`: Sampling temperature. Use `0` for greedy decoding.
- `--top-k`: Top-k sampling value.
- `--top-p`: Top-p sampling value. `0` disables top-p.
- `--cfg-coef`: Classifier-free guidance coefficient.
- `--seed`: Optional random seed.
- `--device`: `auto`, `cuda`, `cpu`, or `mps`.

## Model Downloads and Caching

Model weights are downloaded lazily by AudioCraft and Hugging Face on first run. The scripts mount:

- `~/.cache/huggingface`
- `~/.cache/torch`

Subsequent runs reuse those caches.

Sample inputs are downloaded lazily from `zzsi/cvl` into `~/.cache/huggingface/cvl_data/audiogen/`.

## Attribution and License

This example uses Meta AudioCraft AudioGen from `facebookresearch/audiocraft`, pinned to commit `896ec7c`.

Attribution:

- AudioCraft and AudioGen: Meta Platforms, Inc. and affiliates, FAIR team of Meta AI.
- Source repository: https://github.com/facebookresearch/audiocraft
- Model: https://huggingface.co/facebook/audiogen-medium
- Sample audio asset: `assets/sirens_and_a_humming_engine_approach_and_pass.mp3` from the upstream AudioCraft repository, mirrored at `zzsi/cvl` for this CVL example.
- Paper: "AudioGen: Textually Guided Audio Generation", https://arxiv.org/abs/2209.15352

License:

- AudioCraft source code is MIT licensed.
- Released AudioGen model weights are CC-BY-NC 4.0. Treat this example as research/internal-demo oriented unless model-weight licensing has been reviewed for the intended use.
- Do not imply sponsorship, endorsement, or approval by Meta.

## Notes

AudioCraft's published requirements pin an older Torch stack, so the Dockerfile installs runtime dependencies explicitly and installs AudioCraft with `--no-deps` to keep the PyTorch 2.9.1/CUDA 12.8 base image.

## References

- AudioCraft: https://github.com/facebookresearch/audiocraft
- AudioGen documentation: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md
- AudioGen model: https://huggingface.co/facebook/audiogen-medium
