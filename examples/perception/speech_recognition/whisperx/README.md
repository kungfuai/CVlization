# WhisperX

Speech transcription with WhisperX: VAD-based batching, faster-whisper ASR, forced alignment, word-level timestamps, and optional speaker diarization.

## What To Expect

- First run downloads the `tiny.en` Whisper model by default, the English alignment model, NLTK Punkt data, and a shared CVL sample WAV from `zzsi/cvl`. Expect about 450 MB of cached model/sample data for the smoke path. Larger Whisper models can add several GB.
- The default command transcribes the shared sample audio, runs forced alignment, and writes `whisperx_transcript.json` in your current working directory when run through `cvl run`.
- JSON output contains the transcript, segment timestamps, word-level timestamps, language, and run metadata. `txt`, `srt`, and `vtt` outputs are also supported.
- CPU smoke tests use `tiny.en` with int8 compute. Verified runtime was about 24 seconds on first run and about 7 seconds with cached data on an Apple M5 Pro host running the Linux AMD64 image under Docker emulation.
- Speaker diarization is opt-in because it requires a Hugging Face token and access to the gated pyannote diarization model.

## Sample

**Input** - shared sample audio:

```text
https://huggingface.co/datasets/zzsi/cvl/resolve/main/livetalk/example.wav
```

**Output excerpt** - expected `tiny.en` JSON content:

```json
{
  "text": "It's an amazing gift and a unique privilege and I love going",
  "language": "en",
  "aligned": true,
  "diarized": false,
  "model": "tiny.en"
}
```

The exact transcript and word timestamps depend on the selected model and compute type. Use a larger model for better accuracy.

## Build

```bash
cvl run whisperx build
```

or:

```bash
cd examples/perception/speech_recognition/whisperx
./build.sh
```

## Run

Default CPU-friendly sample transcription with alignment:

```bash
cvl run whisperx predict
```

Transcribe your own audio:

```bash
cvl run whisperx predict -- --audio meeting.wav --output meeting.json
```

Write subtitles:

```bash
cvl run whisperx predict -- \
  --audio lecture.wav \
  --model small \
  --format srt \
  --output lecture.srt
```

Use a larger model on an NVIDIA GPU:

```bash
CVL_USE_GPU=1 cvl run whisperx predict -- \
  --audio meeting.wav \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --batch-size 16 \
  --output meeting.json
```

Enable speaker diarization:

```bash
HF_TOKEN=hf_xxx cvl run whisperx predict -- \
  --audio panel.wav \
  --model large-v3 \
  --diarize \
  --min-speakers 2 \
  --max-speakers 4 \
  --output panel.json
```

## Options

Common flags:

- `--audio`: audio file path, URL, or `sample`.
- `--model`: Whisper model size/name. Default: `tiny.en`.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--compute-type`: CTranslate2 compute type. Defaults to `float16` on CUDA and `int8` on CPU.
- `--no-align`: skip forced alignment.
- `--diarize`: enable speaker diarization. Requires `HF_TOKEN` or `--hf-token`.
- `--format`: `json`, `txt`, `srt`, or `vtt`.

## Model Downloads And Caching

Models and sample audio are downloaded lazily and cached under mounted host cache directories:

- `~/.cache/cvlization/whisperx/models` to `/workspace/models`
- `~/.cache/cvlization/huggingface` to `/root/.cache/huggingface`
- `~/.cache/cvlization/torch` to `/root/.cache/torch`

The default sample audio is cached under `~/.cache/cvlization/huggingface/cvl_data/whisperx/`.

## Notes

- WhisperX uses faster-whisper for ASR, then adds forced alignment and optional diarization.
- GPU inference requires Docker GPU support and compatible NVIDIA drivers.
- On Apple Silicon this example runs on CPU because WhisperX/faster-whisper does not expose an MPS path.
- Diarization uses `pyannote/speaker-diarization-community-1`; accept the model terms on Hugging Face before using `--diarize`.
- No sample audio is committed to this repo. Replace the default sample path after the Artemis II audio is hosted in `zzsi/cvl`.

## References

- WhisperX: https://github.com/m-bain/whisperX
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- pyannote.audio: https://github.com/pyannote/pyannote-audio
