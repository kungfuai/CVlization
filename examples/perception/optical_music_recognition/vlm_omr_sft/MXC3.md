# DEPRECATED — MXC3 (2026-05-21)

**Superseded** by a simpler MXC2-based pipeline: per-measure SFT on MXC2
with a separate `image → key=N` classifier and a deterministic re-spell
pass at inference. See task list for the new plan.

The MXC3 decoupling motivation (per-aspect parallel channels for
easier VLM training) was speculative and never validated. The Level 7a
investigation showed the residual error is a wrong-key belief that
cascades through note spellings — that's better addressed by isolating
the key prediction and re-spelling, not by splitting the target into
many channels.

Coverage at deprecation: L7a 100%, L9 100%, openscore lieder ~84%
(round-trip via musical-equivalence normalizer). The encoder/decoder/
slicer code is retained for reference.

---

# MXC3 — Decoupled MusicXML Compact format v3

A token-efficient, channel-decoupled encoding of MusicXML, designed so
per-measure and per-aspect ground truth are trivially extractable.

## Why MXC3 (vs MXC2)

MXC2 bundles position + accidental + key into one absolute-pitch token
(e.g. `D#4`). A wrong key cascades into many wrong pitch tokens.
MXC2 is also stateful (stem/voice/staff/divisions only emitted on
change), which makes per-measure slicing require state reconstruction.

MXC3 fixes both:

1. **Decoupled channels.** Position (`D4`), duration (`q`), stem (`u`),
   accidental override (only when deviating from the active key), and
   any other note attributes are *separate lines* within a measure block.
2. **Self-contained measures.** Every measure block carries its full
   active context (key, time, clef, default stem). Slicing one measure
   is `extract block` — no walking earlier state.
3. **No key baked into pitches.** `pos:` is diatonic letter+octave
   only. The absolute pitch is `apply_key(pos, key, acc_override)` —
   a deterministic post-decode. A wrong key never corrupts positions.

## Scope of v1 — L7a feature set

L7a uses: multi-part (3 staves: Voice + Piano RH + Piano LH), chords,
rests, dots, stems, explicit accidentals. **Not in L7a**: ties, slurs,
lyrics, beams, dynamics, articulations, ornaments, tempo, backup,
forward, multi-voice, multi-staff per part.

MXC3 v1 covers the L7a-needed subset cleanly. Hooks for v2 (lyrics,
ties, slurs, multi-voice) are reserved but not implemented.

## Format

```
HEADER parts=N
P=1 name="<text>"  abbr="<text>"
P=2 name="<text>"  abbr="<text>"
...

M=1 P=1 key=<N> time=<beats>/<beat-type> clef=<sign><line> stem=<default>
  pos: <elem1> <elem2> <elem3> ...
  dur: <dur1> <dur2> <dur3> ...
  acc: <i>=<kind> <j>=<kind> ...    (only when overrides present)
  stem: <i>=<dir> <j>=<dir> ...     (only when overriding measure default)

M=1 P=2 key=<N> time=<beats>/<beat-type> clef=<sign><line> stem=<default>
  pos: ...
  dur: ...
  ...

M=2 P=1 key=<N> time=<beats>/<beat-type> clef=<sign><line> stem=<default>
  ...
```

### Token rules

- **`pos:` element**: either a single diatonic letter+octave (e.g. `D4`)
  or a chord written as bracketed list `[D4,F#4,A4]`. A rest is `R`.
  Wait — *positions never carry accidentals*. So chord is
  `[D4,F4,A4]` (no `#` in pos), with the sharp emitted on the `acc:`
  line when overriding the key. **Decision: positions are strictly
  diatonic letter + octave; never any accidental glyph.**
- **`dur:` element**: a duration name (`whole`, `half`, `quarter`,
  `eighth`, `16th`, `32nd`, `64th`, `128th`) optionally suffixed with
  `.` for dotted (`half.` = dotted half) or `..` for double-dotted.
  Tuples (later): `quarter:3in2` for triplet quarter.
- **`acc: <beat_index>=<kind>`**: explicit accidental override at the
  given beat index in this measure's `pos:` list. `kind` ∈ {`#`, `##`,
  `b`, `bb`, `n`}. Only emitted when the accidental on the note differs
  from what the key implies (or there's an explicit accidental in the
  source MusicXML). Missing index → use key.
- **`stem:`**: measure header has `stem=u|d|n` default. `stem:` line
  lists per-beat overrides `<i>=u|d|n` if any. If all notes share the
  default, omit the `stem:` line.
- **Beat index**: zero-based position in the `pos:` list. Same indexing
  for all aspect lines (parallel arrays).
- **Whitespace**: 2-space indent inside a measure block. Single space
  between tokens. Empty channel lines omitted.

### Header

```
HEADER parts=3 [movement-title="..."] [composer="..."] [work-title="..."]
P=1 name="Voice"
P=2 name="Piano RH"
P=3 name="Piano LH"
```

### Measure block — minimum required fields

Every `M=N P=K` line must include: `key=<int>`, `time=<beats>/<beat>`,
`clef=<sign><line>`. (Even if unchanged from previous measure — the
point is self-containment.) Optional: `stem=<u|d|n>` if there's a
measure-wide default; otherwise each note's stem must appear in
`stem:`.

### Future hooks (reserved syntax)

- `tie: <i>=start <j>=stop ...` — tie endpoints by beat index.
- `slur: <num>:<i>=start <num>:<j>=stop ...`
- `lyr1: <syl> -- <syl> ...` — aligned with `pos:` by index. `--` for no syllable.
- `voice:` sub-blocks for multi-voice measures: `M=5 P=1 V=2 ...`
- `staff:` sub-blocks for multi-staff parts: `M=5 P=1 S=2 ...`
- Direction / dynamics: separate `dir:` line per measure.
- Barline styling: `bar=<style> [loc=<left|right>] ...`

## Reversibility

The encoder `xml_to_mxc3(xml)` and decoder `mxc3_to_xml(mxc3)` must
round-trip for any L7a MusicXML — modulo:
- Whitespace and element ordering (XML-canonical equivalence).
- Stateful redundancy: MusicXML emits `<key>` only when it changes;
  MXC3 emits it every measure. The round-trip drops redundant
  `<attributes>` blocks on decode.

A `roundtrip_mxc3.py` script validates: for every L7a dev sample,
`mxc2(xml) == mxc2(mxc3_to_xml(xml_to_mxc3(xml)))` (i.e. MXC2 of the
round-tripped XML matches MXC2 of the original — uses MXC2 as the
musical-equivalence oracle, which we know works on L7a).

## Slicers

```python
extract_measure(mxc3, m, p)            # one measure block
extract_aspect(mxc3, m, p, "pos")      # one aspect line
extract_header(mxc3)                   # just the header (key, parts, etc.)
```

Each is a regex-shaped operation over the flat MXC3 text.

## Example — measure 1 of a +3 page in MXC3 v1

```
HEADER parts=3
P=1 name="Voice"
P=2 name="Piano RH"
P=3 name="Piano LH"

M=1 P=1 key=3 time=4/4 clef=G2 stem=u
  pos: A4 B4 A4
  dur: half quarter quarter

M=1 P=2 key=3 time=4/4 clef=G2 stem=d
  pos: [A3,C4,E4] [G3,B3,D4] [A3,C4,E4]
  dur: quarter quarter quarter

M=1 P=3 key=3 time=4/4 clef=F4 stem=d
  pos: A2 C3
  dur: half half
```

The `pos:` line has diatonic letters only. With `key=3`, applying the
key gives `C#`/`F#`/`G#` automatically. If a particular note's
accidental deviates (e.g. explicit `F` natural where key implies F#),
emit `acc: <index>=n` on that measure.

## Open design questions (defer to commitment)

These are deferred until v2 work; not blockers for L7a v1:

1. How to encode tuplets at the `dur:` level cleanly (currently:
   `quarter:3in2`).
2. How to handle voice splits when a part uses multiple voices on the
   same staff (multi-voice openscore content).
3. Direction/text encoding policy (escape rules for free-form text).
