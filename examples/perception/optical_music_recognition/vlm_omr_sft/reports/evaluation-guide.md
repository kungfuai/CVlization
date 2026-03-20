# Evaluating OMR Inference Quality

Guide for vetting the quality and accuracy of model inference output,
whether in MXC or XML format.

## Quick checks (eyeball)

### 1. Does the output parse?

- **XML**: Does `xml.etree.ElementTree.fromstring()` succeed?
- **MXC**: Does `mxc_to_xml(prediction)` succeed?

If parsing fails, the model hasn't learned the format syntax yet.

### 2. Does it contain actual notes?

```python
# MXC
note_lines = [l for l in pred.split('\n') if l.strip().startswith('N ')]
rest_lines = [l for l in pred.split('\n') if l.strip().startswith('R ')]
print(f'{len(note_lines)} notes, {len(rest_lines)} rests')
```

```python
# XML
import re
notes = re.findall(r'<note>.*?</note>', pred, re.DOTALL)
pitched = [n for n in notes if '<pitch>' in n]
rests = [n for n in notes if '<rest' in n]
```

If all rests and no pitched notes, the model learned structure but not note reading.

### 3. Pitch variety

```python
import re
pitches = [re.match(r'N (\S+)', l.strip()).group(1)
           for l in pred.split('\n') if re.match(r'\s*N [A-G]', l)]
unique = set(pitches)
print(f'{len(pitches)} notes, {len(unique)} unique pitches')
print(f'Pitches: {sorted(unique)}')
```

| Unique pitches | Assessment |
|---|---|
| 1 | Monotone — model defaulting to a safe pitch |
| 2-5 | Some variety but likely mechanical pattern |
| 6-12 | Emerging pitch discrimination |
| 12+ | Likely reading from image (or hallucinating octaves — check for octave > 8) |

**Watch for hallucinated octaves** (A10, C17, etc.) — these indicate the model is
varying pitch tokens but not grounding them in the image.

### 4. Melodic contour

Do pitches form a plausible melody or are they random/cyclic?

```python
# Extract numeric pitch values for contour analysis
def pitch_to_midi(p):
    """Rough conversion: C4=60, D4=62, etc."""
    step_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    step = p[0]
    rest = p[1:]
    alter = 0
    for sym, val in [('bb', -2), ('b', -1), ('n0', 0), ('##', 2), ('#', 1)]:
        if rest.startswith(sym):
            alter = val
            rest = rest[len(sym):]
            break
    try:
        octave = int(rest)
    except ValueError:
        return None
    if octave > 8:
        return None  # hallucinated
    return (octave + 1) * 12 + step_map.get(step, 0) + alter

midi_vals = [pitch_to_midi(p) for p in pitches]
midi_vals = [m for m in midi_vals if m is not None]

# Check for repeating patterns
if len(set(midi_vals)) == 1:
    print('MONOTONE')
elif len(midi_vals) > 4:
    # Check if it's a simple cycle
    cycle_len = None
    for cl in range(2, min(20, len(midi_vals) // 2)):
        if midi_vals[:cl] == midi_vals[cl:2*cl]:
            cycle_len = cl
            break
    if cycle_len:
        print(f'CYCLIC pattern of length {cycle_len}')
    else:
        print('VARIED (potentially reading from image)')
```

### 5. Rhythm variety

```python
# MXC: extract type tokens
import re
types = re.findall(r'N \S+ (\S+)', pred)
from collections import Counter
print(Counter(types))
```

Good: mix of `q`, `e`, `h`, `w`, `s` with `dot` modifiers.
Bad: all same type (e.g., all `q 10080`).

### 6. Lyrics alignment

If the score has lyrics, check:
- Are lyrics present in the output? (`L1:s:...` in MXC)
- Do syllabic markers make sense? (`b`=begin, `m`=middle, `e`=end, `s`=single)
- Is the text coherent? (actual words vs garbage)
- Multiple verses? (`L1:...` and `L2:...` on the same note)

### 7. Structure sanity

- **Part count**: Does the prediction have the right number of parts? (Voice + Piano = P1 + P2)
- **Measure count**: Roughly matches the number of measures visible on the page?
- **Key/time/clef**: Compare first measure attributes against what's visible in the image
- **Barlines**: Present at expected locations?

## Comparing against reference

### Token-level comparison (quick)

```python
# Compare reference and prediction character counts
ref_mxc = xml_to_mxc(strip_musicxml_header(sample['musicxml']))
print(f'Reference: {len(ref_mxc)} chars')
print(f'Prediction: {len(pred)} chars')
print(f'Ratio: {len(pred)/len(ref_mxc):.2f}')
```

### Note-level comparison (detailed)

```python
def extract_note_sequence(mxc_text):
    """Extract (pitch, type, duration) tuples from MXC."""
    notes = []
    for line in mxc_text.split('\n'):
        line = line.strip()
        m = re.match(r'N (\S+) (\S+) (\d+)', line)
        if m:
            notes.append((m.group(1), m.group(2), m.group(3)))
        elif line.startswith('R '):
            parts = line.split()
            rtype = parts[1] if len(parts) > 1 else '?'
            rdur = parts[2] if len(parts) > 2 else '?'
            notes.append(('rest', rtype, rdur))
    return notes

ref_notes = extract_note_sequence(ref_mxc)
pred_notes = extract_note_sequence(pred)

# Compare lengths
print(f'Reference: {len(ref_notes)} notes')
print(f'Prediction: {len(pred_notes)} notes')

# Pitch accuracy (for matching note positions)
n = min(len(ref_notes), len(pred_notes))
pitch_match = sum(1 for r, p in zip(ref_notes[:n], pred_notes[:n]) if r[0] == p[0])
type_match = sum(1 for r, p in zip(ref_notes[:n], pred_notes[:n]) if r[1] == p[1])
print(f'Pitch accuracy: {pitch_match}/{n} ({100*pitch_match/n:.0f}%)') if n else None
print(f'Type accuracy: {type_match}/{n} ({100*type_match/n:.0f}%)') if n else None
```

### Metadata accuracy

Compare against what's visible on the page image:
- Title: exact match?
- Composer: correct?
- Key signature: correct number of sharps/flats?
- Time signature: correct beats/beat-type?
- Tempo marking: correct text?

## Standardized evaluation with eval_mxc.py

Use `eval_mxc.py` for consistent, alignment-aware metrics across runs:

```bash
# Evaluate latest WandB run (shows latest step + trend)
python eval_mxc.py --latest

# Evaluate a specific run
python eval_mxc.py --wandb-run wandb/run-20260319_201235-3h8sdtea

# Show per-step summary
python eval_mxc.py --wandb-run wandb/run-XXXXX --by-step

# Evaluate specific step only
python eval_mxc.py --wandb-run wandb/run-XXXXX --step 1400

# Compare two MXC files directly
python eval_mxc.py --pred prediction.mxc --ref reference.mxc
```

### Metrics

All metrics use `SequenceMatcher` for alignment-aware comparison (handles
insertions/deletions without cascading positional mismatches):

| Metric | What it measures |
|---|---|
| **Pitch similarity** | Are the right notes in the right order? |
| **Rhythm similarity** | Are the durations/types correct? |
| **Combined similarity** | Pitch + type together |
| **Note coverage** | pred events / ref events |
| **Longest pitch match** | Longest consecutive correct run |
| **Unique pitches** | Pitch variety (pred vs ref) |
| **Header accuracy** | Key, time signature, part structure |

### Benchmark results (as of 2026-03-20)

| Run | Avg pitch sim | Avg rhythm sim | Coverage | Unique pitches |
|---|---|---|---|---|
| Ministral-3 r=8 MXC ep2 | 7% | 10% | 35% | 0.8 |
| Qwen3.5-9B MXC ep2 | 31% | 32% | 34% | 14.5 |
| Qwen3.5-9B MXC ep4 | 32% | 30% | 33% | 13.8 |
| Qwen3.5-9B MXC ep5+ (4096 tok) | 26% | 28% | 61% | 11.0 |

### Interpreting results

- **Pitch similarity < 10%**: model not reading pitch from image (monotone/random)
- **Pitch similarity 20-40%**: model reading some pitches, with alignment drift
- **Pitch similarity > 50%**: good pitch reading (check longest match for consistency)
- **Coverage > 100%**: model hallucinating extra notes
- **Coverage < 30%**: output token budget too small (increase `inference_max_new_tokens`)

## Red flags

- **Hallucinated octaves** (A10, C17): model is varying tokens but not grounding in image
- **Descending/ascending scale loops**: learned a scale pattern, not reading the image
- **All same duration**: collapsed to a single rhythm
- **Piano part all rests**: model only learned voice part
- **Measure count >> visible measures**: hallucinating extra measures
- **Lyrics in wrong language**: copying from a different training sample
