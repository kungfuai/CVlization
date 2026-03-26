# Training Run: Lieder 2-epoch (2026-03-14)

**WandB run**: `mzwhkuk8` — https://wandb.ai/zzsi_kungfu/openscore-omr/runs/mzwhkuk8

## Config

| Setting | Value |
|---|---|
| Model | `unsloth/gemma-3-4b-pt` (QLoRA 4-bit) |
| Dataset | `zzsi/openscore` `pages_transcribed`, corpus=`lieder` |
| Train rows | 2,986 (shuffled) |
| Val rows | 193 (HF dev split) |
| Epochs | 2 |
| Total steps | 1,494 |
| Batch size | 1 (grad accum 4, effective 4) |
| Learning rate | 2e-4 (cosine) |
| LoRA r/alpha | 16/16 |
| max_length | 4096 |
| GPU | 1× (CUDA_VISIBLE_DEVICES=1) |
| Runtime | ~204 min |

## Training Loss

Loss decreased steadily over 2 epochs. No obvious plateau by the end, suggesting more epochs or data would continue to help.

## Inference Quality Over Time

Spot-checked sample `lc6211535` (5 Songs from the Chinese Poets, p1, bars 1–41).

**Step 0 (untrained baseline):**
- Output: random internet text ("At the end of the day I am a very happy person", Stack Overflow posts, nested HTML tags)
- No MusicXML structure whatsoever

**Step 200:**
- Valid MusicXML structure with `<divisions>`, `<clef>`, `<time>`, `<key>`, tempo markings
- Wrong musical content: key -1 (F major), 3/4 time, "Allegro" — reference is C major, 2/2, "Lento non troppo"

**Step 500:**
- Still valid structure, still wrong content: C major (correct!), 2/4 time (wrong), "Andantino" (wrong)

**Step 1000:**
- E♭ major (wrong), 3/4 (wrong), "Andante" (wrong)

**Step 1400 (final):**
- E♭ major (wrong), 3/4 (wrong), "Allegretto" (wrong)
- But: note structure is well-formed, durations are plausible (`30240` = 3 quarter notes at divisions=10080), measure comments rendered

**Reference (ground truth):**
- C major (fifths=0), 2/2 cut time, "Lento non troppo"

## Conclusions

- The model learned **MusicXML syntax** well within 2 epochs — output is always valid, well-formed XML with correct tag hierarchy
- The model has **not yet learned to read the image** — key, time signature, and tempo are wrong and vary inconsistently across steps
- This is expected: 2 epochs on ~3k pages is likely insufficient for the vision encoder to drive the decoder toward image-conditioned output
- No mode collapse observed (unlike the unsorted all-corpus run which collapsed to "Symphony No.9")

## Issues Fixed During This Run

- Removed noisy metadata from training targets (`<movement-title>tmp*.xml`, `<identification>`, `<defaults>` blocks) via `strip_musicxml_header()`
- Dataset shuffled to prevent corpus-level mode collapse
- Switched from manual 10% val carve-out to HF `dev` split (193 lieder rows)
- WandB step ordering fixed (removed explicit `step=` from `wandb.log`)
- Step-0 baseline logged via `on_train_begin`

## Next Steps

- Train on all 3 corpora (lieder + quartets + orchestra) with shuffling
- Increase epochs or use `num_train_epochs: 5+`
- Evaluate whether the vision encoder is contributing by comparing loss with vs without image input
- Consider a smaller, cleaner dataset subset for faster iteration
