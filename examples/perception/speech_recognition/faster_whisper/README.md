# Faster-Whisper

Fast speech transcription and translation using `faster-whisper`, a CTranslate2 implementation of OpenAI Whisper models.

## What To Expect

- First run downloads the `tiny.en` CTranslate2 model by default, roughly 75 MB, plus a shared CVL sample WAV from `zzsi/cvl`. Larger models such as `large-v3` download several GB.
- The default command transcribes the shared sample audio and writes `faster_whisper_transcript.json` in your current working directory when run through `cvl run`.
- JSON output contains the transcript, detected language, segment timestamps, and run metadata. `txt`, `srt`, and `vtt` outputs are also supported.
- CPU smoke tests use `tiny.en` with int8 compute. Verified Docker smoke runtime was about 6 seconds on first sample run and about 3 seconds with cached data on an Apple M5 Pro host running the Linux AMD64 image under Docker emulation. GPU runs with larger models are substantially faster.

## Sample

**Input** - shared sample audio:

```text
https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav
```

**Output excerpt** - default `tiny.en` JSON output:

```json
{
  "text": "It's an amazing gift and a unique privilege and I love going.",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 5.0,
      "text": " It's an amazing gift and a unique privilege and I love going."
    }
  ],
  "language": "en",
  "language_probability": 1,
  "duration": 5.0,
  "model": "tiny.en"
}
```

The exact transcript depends on the selected model and compute type. Use a larger model for better accuracy.

## Build

```bash
cvl run faster-whisper build
```

or:

```bash
cd examples/perception/speech_recognition/faster_whisper
./build.sh
```

## Run

Default CPU-friendly sample transcription:

```bash
cvl run faster-whisper predict
```

Transcribe your own audio:

```bash
cvl run faster-whisper predict -- --audio meeting.wav --output meeting.json
```

Use a larger model on GPU:

```bash
CVL_USE_GPU=1 cvl run faster-whisper predict -- \
  --audio meeting.wav \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --vad \
  --word-timestamps \
  --output meeting.json
```

Write subtitles:

```bash
cvl run faster-whisper predict -- \
  --audio lecture.wav \
  --model small \
  --format srt \
  --output lecture.srt
```

Translate non-English speech to English:

```bash
cvl run faster-whisper predict -- \
  --audio talk_ja.wav \
  --model medium \
  --language ja \
  --task translate \
  --format txt \
  --output talk_en.txt
```

## Options

Common flags:

- `--audio`: audio file path, URL, or `sample`.
- `--model`: faster-whisper model size/name or local CTranslate2 model path. Default: `tiny.en`.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--compute-type`: CTranslate2 compute type. Defaults to `float16` on CUDA and `int8` on CPU.
- `--vad`: enable Silero voice activity detection.
- `--word-timestamps`: include word timestamps in JSON output.
- `--format`: `json`, `txt`, `srt`, or `vtt`.

## Model Downloads And Caching

Models are downloaded lazily by `faster-whisper` and cached under the mounted Hugging Face cache. The wrapper mounts:

- `~/.cache/cvlization/huggingface` to `/root/.cache/huggingface`
- `~/.cache/cvlization/torch` to `/root/.cache/torch`

The default sample audio is cached under `~/.cache/cvlization/huggingface/cvl_data/faster_whisper/`.

## Notes

- `faster-whisper` uses PyAV for decoding, so system `ffmpeg` is not required for the normal path.
- GPU inference requires Docker GPU support and compatible NVIDIA drivers.
- `predict.sh` runs without Docker GPU flags by default. Set `CVL_USE_GPU=1` for all GPUs, or `CVL_GPU=0` for one GPU.
- On Apple Silicon Docker runs this example on CPU because CTranslate2 does not use MPS through this wrapper.

## References

- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- OpenAI Whisper: https://github.com/openai/whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2
