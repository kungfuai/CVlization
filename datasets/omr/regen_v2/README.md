# v2 dataset regeneration (SVG → cairosvg → 1280-wide PNG)

One-off helper scripts used to regenerate `zzsi/synthetic-scores` and
`zzsi/openscore` with PNGs rendered via the cairosvg pipeline at a
consistent 1280-px width, replacing the older 150-DPI `lilypond --png`
renderings. Same config names, same schema — only the pixel data
changes. See commit message for context.

## Scripts

- `regen_openscore_cache.py` — rasterizes every `~/.cache/openscores/svg/.../score-*.svg`
  to `~/.cache/openscores_v2/.../score-pageN.png` at width=1280 via cairosvg,
  parallel via `ProcessPoolExecutor`.
- `regen_synth_hf.py` — pulls a synthetic-scores HF split's MusicXML rows,
  renders them via the new `lilypond --svg + cairosvg` pipeline (see
  `synthetic_scores/generate.batch_render`), saves PNGs to
  `~/.cache/synthetic_v2/<config>/<split>/<score_id>.png`.
- `push_synth_v2.py` — streams an existing `zzsi/synthetic-scores` split,
  swaps the `image` column with the v2 PNG (matched by `score_id`), and
  pushes back to the same config + split. Schema columns (musicxml,
  score_id, level) are unchanged.
- `push_openscore_v2.py` — same idea for `zzsi/openscore`. Matches by
  (score_id, page).

## Reproducing

```sh
# Stage 1: openscore (~17k SVGs already cached, fast)
python3 regen_openscore_cache.py --workers 16

# Stage 2: synthetic-scores (re-renders via LilyPond docker)
for LVL in level1 level2 level3 level4 level5 level6 level6b \
          level7 level7a level7b level7c level8 level9; do
  for SPLIT in train dev test; do
    python3 regen_synth_hf.py --config $LVL --split $SPLIT --limit 2000 &
  done
done; wait

# Stage 3: push (serialize within each repo to avoid 409 Conflict)
for LVL in level1 ... level9; do
  for SPLIT in train dev test; do
    python3 push_synth_v2.py --level $LVL --split $SPLIT
  done
done
for CFG in pages_transcribed pages; do
  for SPLIT in train dev test; do
    python3 push_openscore_v2.py --config $CFG --split $SPLIT
  done
done
```

The HF Hub serializes commits per-repo; expect 409 Conflict errors when
running 13+ parallel pushes against the same dataset. Either serialize
within each repo or retry on conflict.
