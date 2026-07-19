# MOSS-Transcribe-Diarize

Joint long-form transcription, timestamps, and speaker diarization in a single
ungated 0.9B-parameter model. No separate pyannote or gated models required.

## Sample

**Input** -- multi-speaker audio (auto-downloaded, ~600 KB):

Two speakers reading excerpts from LibriSpeech (concatenated, 19.6 s total).
Hosted at [`zzsi/cvl/moss_transcribe_diarize/multi_speaker_sample.wav`](https://huggingface.co/datasets/zzsi/cvl/resolve/main/moss_transcribe_diarize/multi_speaker_sample.wav).

**Output** -- timestamped speaker-aware transcript
([sample](https://huggingface.co/datasets/zzsi/cvl/resolve/main/moss_transcribe_diarize/sample_transcript.json)):

```json
{
  "segments": [
    {"start": 0.19, "end": 4.25, "speaker": "S01", "text": "He was in a fevered state of mind, owing to the blight, his wife's action threatened"},
    {"start": 4.25, "end": 6.11, "speaker": "S01", "text": "to cast upon his entire future."},
    {"start": 6.35, "end": 11.31, "speaker": "S02", "text": "She was four years older than i, to be sure, and had seen more of the world."},
    {"start": 11.78, "end": 15.65, "speaker": "S02", "text": "But i was a boy, and she was a girl, and i resented her protecting manner."},
    {"start": 16.83, "end": 19.57, "speaker": "S01", "text": "He was in a fevered state of mind, owing to the blight."}
  ],
  "num_speakers": 2,
  "speakers": ["S01", "S02"]
}
```

## What to Expect

- **First-run cost**: ~2 GB model download on first run (cached afterward).
  Docker image is ~6 GB.
- **What it does**: transcribes audio and jointly assigns speaker labels
  (`[S01]`, `[S02]`, ...) with start/end timestamps per segment.
- **Where output goes**: `moss_transcript.json` in your current directory
  (when using `cvl run`).
- **Output format**: JSON with `segments` (each has `start`, `end`, `speaker`,
  `text`), `speakers` list, plus raw model text.
- **Runtime**: ~14 s cached on RTX PRO 6000 Blackwell (95.6 GiB VRAM).
  Peak VRAM: 2522 MiB process / 2545 MiB device (2.46 / 2.49 GiB), bfloat16.
  Measured with `monitor_vram.sh` at 200 ms polling (68 samples).

## Quick Start

```bash
# Build
cvl run moss-transcribe-diarize build

# Transcribe the default multi-speaker sample
cvl run moss-transcribe-diarize predict

# Transcribe your own audio
cvl run moss-transcribe-diarize predict -i audio=path/to/conversation.wav
```

Or without `cvl`:

```bash
cd examples/perception/speech_recognition/moss_transcribe_diarize
./build.sh
USE_GPU=1 ./predict.sh --audio sample
USE_GPU=1 ./predict.sh --audio /path/to/conversation.wav --output result.json
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | `sample` | Audio file path, URL, or `sample` for the built-in multi-speaker clip |
| `--model` | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | HuggingFace model ID or local path |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--max-new-tokens` | `2048` | Maximum tokens to generate (increase for long audio) |
| `--prompt` | *(English diarization)* | Custom transcription prompt |
| `--output` | `moss_transcript.json` | Output JSON path |

## Model Details

| Property | Value |
|----------|-------|
| **Model** | [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) |
| **Parameters** | 0.9B |
| **Architecture** | Whisper-Medium encoder + Qwen3-0.6B decoder |
| **License** | Apache-2.0 |
| **Input** | Audio (WAV, MP3, FLAC, video containers via PyAV) |
| **Output** | Timestamped speaker-diarized transcript |
| **Sampling rate** | 16 kHz |
| **VRAM** | 2522 MiB process peak (2.46 GiB), bfloat16 |

## References

- [MOSS-Transcribe-Diarize on HuggingFace](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- [MOSS-Transcribe-Diarize on GitHub](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [arXiv: 2601.01554](https://arxiv.org/abs/2601.01554)
