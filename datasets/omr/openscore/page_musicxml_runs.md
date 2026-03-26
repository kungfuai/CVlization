# page_musicxml.py — Run Log

Records of all `page_musicxml.py` runs generating per-page MusicXML for the
`zzsi/openscore` `pages_transcribed` HuggingFace config.

---

## Lieder

**Command:**
```
python page_musicxml.py --corpus lieder --output /tmp/openscore_pages_lieder
```

**Scores**: 1,372 / 1,460 rendered (81 render failures in Docker)

**Results** (saved to `/tmp/openscore_pages_lieder/lieder`):

| Split | Rows |
|-------|------|
| dev   | 209  |
| test  | 223  |
| train | 3,312 |
| **Total** | **3,744** |

**Notes:**
- ~23% of lieder scores have a pickup bar (measure 0); `_load_score()` detects
  and adjusts offset automatically.
- `slice_musicxml` retries with `gatherSpanners=False` on TypeError (music21
  arpeggio spanner bug).

---

## Quartets

**Command:**
```
python page_musicxml.py --corpus quartets --output /tmp/openscore_pages_quartets
```

**Scores**: 122 MXL files indexed

**Docker render failures (4):**
- `Bartók, Béla / String Quartet No.2` — LilyPond syntax error (`\time 3/8 + 4/8`)
- `Elgar / String Quartet Op.83` — Cannot translate zero-duration object
- `Janáček / String Quartet No.1 "Kreutzer Sonata"` — unknown MusicXML type: None
- `Janáček / String Quartet No.2 "Intimate Letters"` — unknown MusicXML type: None

**music21 slice failures (notable):**
- `sq7313978` page 13 — measure already in Stream
- `sq13744399` — expected 78 SVG pages, found 80; page 1 slice failed (inexpressible durations)
- `sq8928855` page 1 — inexpressible durations
- `sq8661306` page 59 — inexpressible durations
- `sq9790696` page 23 — measure already in Stream
- `sq9071492` page 16 — inexpressible durations
- `sq9631717` page 22 — measure already in Stream
- `sq7555331` page 33 — `'3'` (key error)
- `sq8482283` page 23 — inexpressible durations (Violoncello)

**Notable large scores:**
- `sq7354505` — 347 pages, 48 page-number-only pages skipped
- `sq8823783` — 316 pages
- `sq9631717` — 257 pages

**Results** (saved to `/tmp/openscore_pages_quartets/quartets`):

| Split | Rows |
|-------|------|
| dev   | 136  |
| train | 6,274 |
| **Total** | **6,410** |

224 rows skipped (page-number-only pages with no bar numbers).

**Runtime:** ~12–16 hours total (Docker render + music21 slicing).

---

## Orchestra

**Command:**
```
python page_musicxml.py --corpus orchestra --output /tmp/openscore_pages_orchestra
```

**Scores**: 94 MXL files indexed

**Docker render failures (2):**
- `Boulanger, Lili / D'un matin de printemps` — `'NoneType' object has no attribute 'append'`
- `Brahms / Ein Deutsches Requiem Op.45/1` — `Need to define either prefixCompositeMusic or groupedMusicList`

**music21 notes:**
- Massive MIDI channel warnings throughout: `we are out of midi channels! help!`
  (harmless — orchestral scores exceed MIDI's 16-channel limit; MusicXML export unaffected)
- `Beethoven_Op.125_4` (9th Symphony mvt 4): 629 SVG pages, 139 page-number-only skipped
- `Brahms_Op.68_4` (1st Symphony mvt 4): 62 page-number-only pages skipped

**Notable SVG page-count discrepancies:**
- `Beethoven_Op.21_4` — expected 37, found 41
- `Beethoven_Op.92_1` — expected 79, found 81
- `Bruckner_WAB.105_4` — expected 149, found 157
- `Beach_Op.32_1` — expected 93, found 96

**Results** (saved to `/tmp/openscore_pages_orchestra/orchestra`):

| Split | Rows |
|-------|------|
| test  | 242  |
| train | 4,870 |
| **Total** | **5,112** |

258 rows skipped. No dev split produced (orchestra dev scores all had page-count
mismatches or no valid bar numbers).

**Runtime:** ~16–20 hours total (Docker render ≈1h, music21 slicing ≈15–19h).

---

## HuggingFace Push Summary

**Repo:** `zzsi/openscore`

| Config | Description | Splits |
|--------|-------------|--------|
| `scores` | Full MusicXML per score (default config) | train/dev/test |
| `pages` | Page images only, all 3 corpora | train/dev/test |
| `pages_transcribed` | Image + per-page MusicXML, all 3 corpora | train/dev/test |

**pages_transcribed total rows (approx):**

| Split | Lieder | Quartets | Orchestra | Total |
|-------|--------|----------|-----------|-------|
| dev   | 209    | 136      | 0         | ~345  |
| test  | 223    | —        | 242       | ~465  |
| train | 3,312  | 6,274    | 4,870     | ~14,456 |

(Exact numbers may differ after PNG join — rows without a matching rendered PNG
are dropped.)

**Streaming usage:**
```python
from datasets import load_dataset

# Efficient: corpus column is sorted → row-group predicate pushdown works
ds = load_dataset("zzsi/openscore", "pages_transcribed", streaming=True, split="train")
lieder_only = ds.filter(lambda r: r["corpus"] == "lieder")
```

---

## Known Data Quality Issues

Found during MXC round-trip testing on 500 lieder samples (2026-03-19):

### Malformed XML (~0.6% of samples)

3 out of 500 lieder samples contain MusicXML with mismatched tags that fail
`xml.etree.ElementTree.fromstring()`. Known bad score IDs:

- `lc6564767` page 1
- `lc5729178` page 1
- `lc8712648` page 1

**Root cause:** Likely music21 export bugs during `slice_musicxml`. The full-score
MXL is valid, but the per-page slice produces malformed XML.

**Workaround:** Filter at training time — skip samples that fail XML parsing.

**Fix:** Re-run `page_musicxml.py` for these specific scores with a patched music21,
or post-process slices with an XML validator/repairer.

### Non-breaking spaces in lyrics (cosmetic)

~7 out of 27,632 lyric `<text>` elements contain `\xa0` (non-breaking space)
instead of regular spaces. Example: `"1.\xa0Thou"` instead of `"1. Thou"`.

**Workaround:** Normalize `\xa0` → ` ` during preprocessing.

**Fix:** Add a text normalization pass in `page_musicxml.py` after music21 export.
