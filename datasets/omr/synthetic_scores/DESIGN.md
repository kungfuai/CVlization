# Synthetic Score Generator — Design

## Purpose

Generate (MusicXML, PNG) pairs with controlled complexity for OMR model
diagnostics. Each level adds specific musical features so we can isolate
where the model's accuracy breaks down.

## Difficulty Ladder

### Implemented

| Level | New features | Cumulative content | Measures |
|---|---|---|---|
| 1 | Baseline | Single staff, C major, quarter notes only | 8 |
| 2 | + Rhythm variety | Half, eighth, whole, dotted notes | 16 |
| 3 | + Key signatures, accidentals | All major/minor keys, chromatic notes (10%) | 16 |
| 4 | + Rests, ties | Rest patterns, tied notes across beats | 16 |
| 5 | + Grand staff | Two staves (treble+bass), monophonic per hand | 16 |
| 6 | + Chords | Homophonic chords in RH (2-3 note voicings) | 16 |
| 7 | + Voice part, lyrics | Voice + piano + English syllables | 24 |

### Planned

| Level | New features | Cumulative content | Measures |
|---|---|---|---|
| 8 | + Beams, ties, dynamics, directions | All Level 7 content + beam groups, tied notes, dynamic markings (p, f, mf, cresc.), tempo/expression text | 24 |
| 9 | + Complex rhythms | All Level 8 + triplets, syncopation, grace notes, 16th-note runs | 16-24 |
| 10 | + Dense polyphony | All Level 9 + multi-voice per staff (voice 1 + voice 2 on same staff), cross-staff notation | 8-16 |

## Level 8 Design (next to implement)

### Motivation

OLiMPiC real piano data achieves only 39% pitch accuracy despite no token
truncation and 15K training samples. Comparing OLiMPiC MXC to Level 7 MXC
reveals specific features present in real music but absent from synthetic:

| Feature | Level 7 | OLiMPiC | Level 8 (planned) |
|---|---|---|---|
| Beams (bb/be/bc) | No | Yes | **Yes** |
| Ties (tie=start/stop) | No | Yes | **Yes** |
| Dynamics (p, f, mf) | No | Some | **Yes** |
| Directions (tempo, expr.) | No | Some | **Yes** |
| Triplets | No | Some | No (Level 9) |
| Multi-voice per staff | No | Some | No (Level 10) |

### Specification

**Parts**: Voice + Piano (same as Level 7)

**Measures**: 24

**Rhythm patterns**: Same as Level 7, but with explicit beam groups:
- Consecutive 8th notes → beam group (bb on first, bc on middle, be on last)
- Dotted quarter + eighth → beam the eighth to next note if applicable

**Ties**: ~10% of notes tied to the next note of the same pitch.
- Insert `tie="start"` on the first note, `tie="stop"` on the second
- Only tie when next note has the same pitch (musically valid)

**Dynamics**: One dynamic marking every 4-8 measures, randomly chosen from:
- pp, p, mp, mf, f, ff
- Placed as `<dynamics>` in a `<direction>` element below the staff

**Directions**: One expression/tempo text per piece:
- Tempo: "Andante", "Allegro", "Moderato", "Adagio", "Lento"
- Placed at measure 1 as `<words>` in a `<direction>` element

**Everything else**: Inherited from Level 7 (key signatures, accidentals,
chords in piano RH, lyrics on voice part, bass clef LH).

### Expected outcome

If Level 8 accuracy is similar to Level 7 (~47%), then beams/ties/dynamics
are not the bottleneck — the complexity limit is elsewhere (chord density,
page length, pitch range).

If Level 8 accuracy drops significantly (<30%), then the added notation
features are confusing the model and the training data needs to cover them.

## Observations from experiments

### What the synthetic ladder has shown

1. **Levels 1-2 (100%\*)**: The model can read pitch perfectly on simple scores.
   Confirms the vision encoder works for basic pitch discrimination.

2. **Level 3 (95%\*)**: Key signature interpretation is a real weakness.
   D→D# errors in A major account for most mistakes.

3. **Level 7 (47%, n=50)**: Adding voice+piano+chords+lyrics drops accuracy
   significantly. Token truncation is a confounder (MXC ~4200 tokens, ceiling 4096).

4. **OLiMPiC (39%\*, n=10)**: Real piano systems with no truncation, 15K samples,
   still only 39%. The ceiling is not data quantity or truncation — it's
   intrinsic notation complexity.

\* n=10, not validated with larger evaluation set.

### Gaps in the current ladder

- **Levels 4-6 are not reliably measured** (n=10 each). Need n=50 evals.
- **No level tests beams, ties, or dynamics** — features present in all real music.
- **No level tests irregular rhythms** (triplets, syncopation) — common in real music.
- **Level 7 has a truncation confounder** — hard to separate accuracy from
  output length limitations.
