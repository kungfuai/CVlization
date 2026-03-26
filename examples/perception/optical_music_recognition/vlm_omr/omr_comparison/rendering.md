# Ekern → PNG Rendering Notes

Notes on rendering ekern/bekern transcriptions to sheet music images via
**verovio** + **cairosvg**. See `render_ekern.py` for the working implementation.

---

## Pipeline

```
ekern/bekern (JSON)
  → strip code fences, replace **ekern_1.0 → **kern, strip @ and ·
  → if no **kern anywhere → prepend **kern\t**kern header (SMT raw bekern)
  → sanitize spine-split sections (*^ / *v) — verovio crashes on them
  → verovio.toolkit().loadData() → renderToSVG()
  → cairosvg svg2png()
  → PNG
```

Run each model in a **subprocess** to isolate verovio C++ crashes from the
main process.

The updated `ekern_transcription` prompt asks the model to output a **complete
kern file** with no code fences. The file may start with `!!!OTL:`/`!!!COM:`
humdrum header records before `**kern` — this is valid and verovio renders them
when `"header": "encoded"` is set.

---

## Title, composer, section labels

Add humdrum header records **before** the `**kern` line:

```
!!!OTL:Biddle's Piano Waltz.
!!!COM:Composed by Robert D. Biddle.
**kern	**kern
...
```

Requires `"header": "encoded"` in verovio options. Verovio renders OTL as
title and COM as composer automatically.

Section labels (INTRO., WALTZ) go inline before the relevant barline:

```
*>INTRO.
=1	=1
```

Verovio renders these as section labels above the staff.

---

## Font — biggest single visual change

```python
tk.setOptions({"font": "Petaluma"})
```

| Font | Character |
|------|-----------|
| `Leipzig` | Default — clean, modern |
| `Bravura` | SMuFL reference, very clean |
| `Petaluma` | **Handwritten/engraved feel — closest to vintage** |
| `Gootville` | Older Lilypond-style |
| `Lassus` | Early music style |

---

## Visual weight — stems, barlines, staff lines

```python
tk.setOptions({
    "staffLineWidth": 0.15,   # default ~0.2; thinner = more vintage
    "barLineWidth":   0.4,    # default ~0.5
    "beamMaxSlope":   15,     # flatter beams
    "spacingStaff":   8,      # gap between staves within a system
    "spacingSystem":  12,     # gap between systems
})
```

---

## Piano brace

Verovio draws the grand staff brace automatically for two-staff kern — no
configuration needed. Requires `*clefF4` / `*clefG2` to be present in the kern.

---

## Post-processing: sepia / aged paper (PIL)

Apply after rendering the PNG, not in verovio:

```python
from PIL import Image, ImageEnhance

img = Image.open("rendered.png").convert("RGB")
# Sepia tone
r, g, b = img.split()
r = r.point(lambda x: min(255, int(x * 1.1)))
b = b.point(lambda x: int(x * 0.85))
img = Image.merge("RGB", (r, g, b))
# Slight fade (aged paper)
img = ImageEnhance.Brightness(img).enhance(0.92)
img.save("rendered_vintage.png")
```

For stronger texture: blend with a scanned paper texture image using PIL's
`Image.paste()` with an inverted-music mask (same technique as SMT's
`SynthGenerator.py`).

---

## What verovio cannot render from VLM/SMT outputs

These require explicit kern encoding that none of the current models produce:

- **Grace notes / ornament glyphs** — models don't output kern grace note syntax
- **Dynamics text** (`ben marcato`, `cres.`, `pp`) — need `!!LO:DY` annotations
- **Glissando lines** — need `\gliss` kern tokens
- **Tremolo** — need `*tremolo` interpretation

---

## Observed issues

| Model | Issue | Effect |
|-------|-------|--------|
| SMT | Spine-split tokens (`*^`/`*v`) | verovio crash → strip before rendering |
| SMT | 41 measures from 1-page input (~30 visible) | Hallucinated repeats/extensions → 6 rendered pages |
| Claude Opus | `2GG 2G` in 3/4 bar (4 beats) | Rhythm error → forced system break → 2 pages |
| Gemini 3.1 Pro | `*k[f#]` (G major) vs correct F major | Wrong key from measure 1 |
| All models | Missing intro section | None transcribed the Andante intro |
