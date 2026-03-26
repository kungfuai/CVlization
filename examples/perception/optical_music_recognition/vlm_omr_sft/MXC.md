# MXC — MusicXML Compact

A line-based compact encoding of MusicXML for training VLM-based Optical Music
Recognition (OMR) models. Achieves ~12x token compression over cleaned MusicXML
while preserving all page-visible musical content through lossless round-trip
conversion.

## Motivation

MusicXML is extremely verbose. A typical lieder page produces ~50K characters
(~14K tokens) of cleaned XML — far beyond the 4096-token training context window.
Even with aggressive stripping of non-visible metadata, only ~10% of pages fit.

The root cause is XML's structural overhead: nested tags, closing tags, indentation,
and verbose element names. A single quarter note:

```xml
<note>
  <pitch>
    <step>C</step>
    <octave>4</octave>
  </pitch>
  <duration>10080</duration>
  <type>quarter</type>
  <stem>up</stem>
  <lyric number="1">
    <syllabic>single</syllabic>
    <text>Lord,</text>
  </lyric>
</note>
```

becomes one MXC line:

```
N C4 q 10080 su L1:s:Lord,
```

~250 chars → ~30 chars. **~8x compression per note**, and the savings compound
across hundreds of notes per page.

## Measured compression

Tested on 500 lieder pages from `zzsi/openscore` `pages_transcribed`:

| Metric | Cleaned XML | MXC |
|--------|-------------|-----|
| Median chars | 50,272 | 4,172 |
| Median ~tokens (÷3.5) | ~14,363 | **~1,192** |
| Compression ratio | — | **12x** |

Most pages fit comfortably within a 4096-token context window with MXC.

## Format specification

### Header section

```
header work-title="Just for Today"
header composer="Jane Bingham Abbott"
header lyricist="Samuel Wilberforce"
header movement-number=1
header movement-title="The Old Fisherman"
```

### Part list

```
P1 Voice Ob
P2 Piano Pno
```

Format: `{part-id} {part-name} {part-abbreviation}`

### Part separator

```
---
```

### Part and measures

```
P1
M 1 div=10080 key=-1 time=3/4 clef=G2
M 2
M 3 clef=F4 clef2=G2 staves=2
```

`M {number}` starts a measure. Attributes are inlined as key=value pairs:

| MXC | MusicXML |
|-----|----------|
| `div=N` | `<divisions>N</divisions>` |
| `key=N` | `<key><fifths>N</fifths></key>` |
| `time=B/T` | `<time><beats>B</beats><beat-type>T</beat-type></time>` |
| `clef=G2` | `<clef><sign>G</sign><line>2</line></clef>` |
| `clef2=F4` | `<clef number="2"><sign>F</sign><line>4</line></clef>` |
| `staves=N` | `<staves>N</staves>` |

### Notes

```
N C4 q 10080 su L1:s:Lord,
N Bb3 e 5040 acc=flat su bm=begin L1:s:for
+N E4 q 10080                        (chord — same onset as previous)
gN D5 e 5040 su                      (grace note)
R q 10080                            (rest)
R whole 30240                        (whole-measure rest)
```

Token order: `[+][g]N|R {pitch} {type} {duration} [modifiers...] [lyrics...]`

**Pitch encoding:** `{step}[alter]{octave}` — `C4`, `Bb3` (flat), `F#5` (sharp), `Cn04` (alter=0, natural)

**Type shorthand:**

| MXC | MusicXML |
|-----|----------|
| `w` | whole |
| `h` | half |
| `q` | quarter |
| `e` | eighth |
| `s` | 16th |
| `t` | 32nd |
| `x` | 64th |
| `bv` | breve |

**Modifiers (order-independent):**

| MXC | Meaning |
|-----|---------|
| `su` / `sd` / `sn` | stem up / down / none |
| `bm=begin` | beam number 1 begin (also `continue`, `end`, `forward-hook`, `backward-hook`) |
| `bm2=begin` | beam number 2 |
| `acc=natural` | visual accidental |
| `dot` / `dot=2` | dotted / double-dotted |
| `tie=start` / `tie=stop` | tie |
| `tied=start` | notational tied (in `<notations>`) |
| `slur1=start` | slur |
| `fermata` | fermata |
| `art=staccato` | articulation |
| `orn=trill-mark` | ornament |
| `v=2` | voice |
| `st=2` | staff |

**Lyrics:** `L{number}:{syllabic}:{text}` — syllabic: `s`=single, `b`=begin, `m`=middle, `e`=end. Spaces in text replaced with `_`.

### Directions

```
dir @above [font-size=12,font-weight=bold] Andante
dir @below dyn=pp
dir @above wedge=crescendo
```

### Barlines

```
bar=light-light
bar=light-heavy loc=left repeat=backward
bar=light-heavy ending=1:start
```

### Navigation

```
bak 10080          (backup — move cursor backward)
fwd 5040 v=2 st=1  (forward — advance cursor with voice/staff)
```

### Layout

```
print new-system
print new-page
```

### Unknown elements

```
xml{<raw-element>content</raw-element>}
```

Generic escape for any MusicXML element not explicitly handled. Preserves
losslessness but defeats compression for that element.

## Implementation

**Module:** `mxc.py` (no external dependencies beyond Python stdlib)

**Public API:**

```python
from mxc import xml_to_mxc, mxc_to_xml

# Compress: cleaned MusicXML → MXC
mxc_text = xml_to_mxc(stripped_xml)

# Expand: MXC → valid MusicXML
xml_text = mxc_to_xml(mxc_text)
```

The input to `xml_to_mxc()` should be post-`strip_musicxml_header()` XML — i.e.,
with non-visible metadata already removed. The output of `mxc_to_xml()` is a valid
`<score-partwise>` document renderable by MuseScore or LilyPond.

## Round-trip accuracy

Tested on 500 lieder pages from the training dataset:

- **496/500 passed** (99.2%)
- 3 failures: malformed XML in source data (not MXC bugs)
- 1 failure: double-space in lyric text (cosmetic)

The round-trip comparison checks: part structure, measure counts, attributes
(divisions, key, time, clef), and per-note fields (pitch, duration, type, stem,
beams, ties, accidentals, dots, voice, staff, lyrics).

## What is preserved vs stripped

The full pipeline is: raw MusicXML → `strip_musicxml_header()` → `xml_to_mxc()`.

**Preserved (visible on page):**

- Work title, composer, lyricist, movement title/number
- Part names and abbreviations
- Key signatures, time signatures, clefs
- Notes: pitch, duration, type, accidentals, dots
- Stems, beams, ties, slurs
- Lyrics (all verses)
- Dynamics, tempo text, other directions with visible text
- Barlines (style, repeats, endings)
- Articulations, ornaments, fermatas
- Voice and staff assignments

**Stripped by `strip_musicxml_header()` (not visible on page):**

- XML declaration, DOCTYPE
- `<defaults>` (page layout / scaling)
- `<encoding>` (software versions)
- `<rights>`, `<creator type="arranger">` (IMSLP metadata)
- `<score-instrument>`, `<midi-instrument>`, `<midi-device>` (playback)
- `<sound tempo=...>` (invisible numeric metadata)
- XML comments
- Empty `<direction>` blocks (only `<sound>` or empty `<words/>`)
- `implicit="no"` attribute (always "no")

## Testing

```bash
# Unit tests (28 forward + round-trip + compression tests)
python -m pytest test_mxc.py -v

# Strip function tests (27 tests)
python -m pytest test_strip_musicxml.py -v

# Real-data round-trip on 500 samples from zzsi/openscore
python test_mxc_realdata.py
```

## Usage in training

To use MXC as training targets, set `target_format: "mxc"` in the config YAML
(not yet integrated — requires updating `convert_to_conversation()` in `train.py`).
The model learns to generate MXC, and predictions are expanded back to MusicXML
via `mxc_to_xml()` for evaluation and rendering.
