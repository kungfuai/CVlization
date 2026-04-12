#!/usr/bin/env python3
"""
Generate synthetic single-page music scores for OMR training diagnostics.

Each score is a single page with controlled complexity. The output is
(MusicXML, PNG image) pairs — no pagination needed.

Difficulty levels:
  1: Single staff, C major, quarter notes only, no rests
  2: Add varied rhythms (half, eighth, whole, dotted)
  3: Add accidentals and key signatures
  4: Add rests, ties, dynamics
  5: Two staves (piano grand staff), monophonic per hand
  6: Add chords (homophonic)
  7: Add lyrics (one verse)
  8: Full lieder complexity (voice + piano, lyrics, dynamics)

Usage:
    python generate.py --level 1 --count 100 --output /tmp/synthetic_scores
    python generate.py --level 1 --count 5 --output /tmp/synthetic_scores --render
"""

import argparse
import os
import random
import subprocess
import tempfile
from pathlib import Path


def _pitch_to_index(pitch_str):
    """Convert 'C4' to a numeric index for interval computation."""
    step = pitch_str[0]
    octave = int(pitch_str[-1])
    return octave * 7 + "CDEFGAB".index(step)


def _index_to_pitch(idx):
    """Convert numeric index back to 'C4' style pitch string."""
    octave = idx // 7
    step = "CDEFGAB"[idx % 7]
    return f"{step}{octave}"


# Pitch pool for Level 1: treble clef range
TREBLE_PITCHES = [
    "A3", "B3",
    "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5",
    "A5",
]
TREBLE_MIN = _pitch_to_index("A3")
TREBLE_MAX = _pitch_to_index("A5")


def _generate_melody(rng, n_notes, pitches=TREBLE_PITCHES):
    """Generate a melody using stepwise motion with occasional leaps.

    Produces musically plausible pitch sequences: mostly steps (±1-2),
    occasional leaps (±3-4), rare large leaps (±5-7).
    """
    pitch_indices = [_pitch_to_index(p) for p in pitches]
    min_idx, max_idx = min(pitch_indices), max(pitch_indices)

    # Start on a random pitch
    current = rng.choice(pitch_indices)
    melody = [current]

    mid_idx = (min_idx + max_idx) // 2
    for _ in range(n_notes - 1):
        # Bias direction toward center when near edges
        dist_from_center = current - mid_idx
        if abs(dist_from_center) > (max_idx - min_idx) // 3:
            # Near edge: bias toward center
            bias = -1 if dist_from_center > 0 else 1
        else:
            bias = rng.choice([-1, 1])

        # Weighted interval size: mostly steps, some leaps
        r = rng.random()
        if r < 0.5:
            mag = 1       # step
        elif r < 0.75:
            mag = 2       # third
        elif r < 0.9:
            mag = 3       # fourth
        else:
            mag = rng.choice([4, 5])  # larger leap

        interval = bias * mag
        new_idx = current + interval
        new_idx = max(min_idx, min(max_idx, new_idx))
        current = new_idx
        melody.append(current)

    return [_index_to_pitch(idx) for idx in melody]


def generate_level1(seed: int, n_measures: int = 8) -> str:
    """Single staff, C major, quarter notes only, no rests.

    Uses real melodic motion (stepwise with occasional leaps) instead of
    random pitch selection. Tests: can the model read pitch from staff position?
    """
    rng = random.Random(seed)
    divisions = 1  # quarter note = 1 division
    n_notes = n_measures * 4
    melody = _generate_melody(rng, n_notes)

    measures = []
    note_idx = 0
    for m in range(1, n_measures + 1):
        notes = []
        for beat in range(4):  # 4/4 time
            pitch = melody[note_idx]
            note_idx += 1
            step = pitch[0]
            octave = pitch[-1]
            midi_approx = _pitch_to_index(pitch)
            stem = "up" if midi_approx < _pitch_to_index("B4") else "down"
            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>
          <octave>{octave}</octave>
        </pitch>
        <duration>{divisions}</duration>
        <type>quarter</type>
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""

        measure_xml = f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>"""
        measures.append(measure_xml)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Treble</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(measures)}
  </part>
</score-partwise>"""


def generate_level2(seed: int, n_measures: int = 16) -> str:
    """Single staff, C major, varied rhythms (half, quarter, eighth, dotted), no rests.

    Tests: can the model read both pitch AND rhythm from the image?
    """
    rng = random.Random(seed)
    divisions = 2  # eighth note = 1, quarter = 2, half = 4, dotted quarter = 3

    # Rhythm patterns that fill a 4/4 measure (total = 8 eighth-note units)
    # Each pattern is a list of (duration, type, dotted) tuples
    PATTERNS = [
        # All quarters
        [(2, "quarter", False)] * 4,
        # Half + two quarters
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        # Two quarters + half
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        # Four eighths + two quarters
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        # Two quarters + four eighths
        [(2, "quarter", False)] * 2 + [(1, "eighth", False)] * 4,
        # Dotted quarter + eighth + half
        [(3, "quarter", True), (1, "eighth", False), (4, "half", False)],
        # Half + dotted quarter + eighth
        [(4, "half", False), (3, "quarter", True), (1, "eighth", False)],
        # Eight eighths
        [(1, "eighth", False)] * 8,
        # Two halves
        [(4, "half", False)] * 2,
        # Dotted half + quarter
        [(6, "half", True), (2, "quarter", False)],
        # Quarter + dotted half
        [(2, "quarter", False), (6, "half", True)],
        # Quarter + two eighths + quarter + two eighths
        [(2, "quarter", False), (1, "eighth", False), (1, "eighth", False),
         (2, "quarter", False), (1, "eighth", False), (1, "eighth", False)],
    ]

    # Generate enough pitches for all measures (max 8 notes per measure)
    all_pitches = _generate_melody(rng, n_measures * 8)
    pitch_idx = 0

    measures = []
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PATTERNS)
        notes = []
        for dur, ntype, dotted in pattern:
            pitch = all_pitches[pitch_idx % len(all_pitches)]
            pitch_idx += 1
            step = pitch[0]
            octave = pitch[-1]
            midi_approx = _pitch_to_index(pitch)
            stem = "up" if midi_approx < _pitch_to_index("B4") else "down"
            dot_xml = "\n        <dot />" if dotted else ""
            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""

        measure_xml = f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>"""
        measures.append(measure_xml)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Treble</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(measures)}
  </part>
</score-partwise>"""


def generate_level3(seed: int, n_measures: int = 16) -> str:
    """Single staff, various key signatures, accidentals, varied rhythms.

    Tests: can the model handle sharps/flats in key signatures and accidentals?
    """
    rng = random.Random(seed)
    divisions = 2

    # Pick a random key signature (-4 to 4 = Ab major to E major)
    fifths = rng.randint(-4, 4)

    # Key signature maps: which notes are sharp/flat
    SHARP_ORDER = "FCGDAEB"
    FLAT_ORDER = "BEADGCF"
    altered = {}
    if fifths > 0:
        for i in range(fifths):
            altered[SHARP_ORDER[i]] = 1  # sharp
    elif fifths < 0:
        for i in range(-fifths):
            altered[FLAT_ORDER[i]] = -1  # flat

    # Pitches: use diatonic scale in the chosen key, plus occasional accidentals
    base_steps = "CDEFGAB"
    pitches_with_alter = []
    for octave in [3, 4, 5]:
        for step in base_steps:
            alter = altered.get(step, 0)
            pitches_with_alter.append((step, octave, alter))

    # Filter to treble clef range
    pitches_with_alter = [(s, o, a) for s, o, a in pitches_with_alter
                          if (o == 3 and s in "AB") or o == 4 or (o == 5 and s in "CDEFGA")]

    # Generate melody using indices
    n_notes = n_measures * 6  # average ~6 notes per measure with varied rhythms
    melody_indices = []
    current = len(pitches_with_alter) // 2  # start in middle
    for _ in range(n_notes):
        melody_indices.append(current)
        # Stepwise motion with centering
        mid = len(pitches_with_alter) // 2
        dist = current - mid
        bias = -1 if dist > len(pitches_with_alter) // 4 else (1 if dist < -len(pitches_with_alter) // 4 else rng.choice([-1, 1]))
        r = rng.random()
        mag = 1 if r < 0.5 else (2 if r < 0.8 else rng.choice([3, 4]))
        new_idx = max(0, min(len(pitches_with_alter) - 1, current + bias * mag))
        current = new_idx

    melody = [pitches_with_alter[i] for i in melody_indices]
    pitch_idx = 0

    # Same rhythm patterns as Level 2
    PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        [(2, "quarter", False)] * 2 + [(1, "eighth", False)] * 4,
        [(3, "quarter", True), (1, "eighth", False), (4, "half", False)],
        [(4, "half", False), (3, "quarter", True), (1, "eighth", False)],
        [(1, "eighth", False)] * 8,
        [(4, "half", False)] * 2,
        [(6, "half", True), (2, "quarter", False)],
        [(2, "quarter", False), (6, "half", True)],
    ]

    measures = []
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PATTERNS)
        notes = []
        for dur, ntype, dotted in pattern:
            step, octave, alter = melody[pitch_idx % len(melody)]
            pitch_idx += 1

            # Occasionally add a chromatic accidental (not in key)
            acc_xml = ""
            alter_xml = ""
            if rng.random() < 0.1:  # 10% chance of accidental
                if alter == 0:
                    chromatic = rng.choice([1, -1])  # sharp or flat
                    alter_xml = f"\n          <alter>{chromatic}</alter>"
                    acc_xml = f"\n        <accidental>{'sharp' if chromatic == 1 else 'flat'}</accidental>"
                else:
                    # Natural accidental (cancel key sig)
                    alter_xml = "\n          <alter>0</alter>"
                    acc_xml = "\n        <accidental>natural</accidental>"
            elif alter != 0:
                alter_xml = f"\n          <alter>{alter}</alter>"

            midi_approx = octave * 7 + "CDEFGAB".index(step)
            stem = "up" if midi_approx < 4 * 7 + 6 else "down"
            dot_xml = "\n        <dot />" if dotted else ""

            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}{acc_xml}
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""

        measure_xml = f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>"""
        measures.append(measure_xml)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Treble</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(measures)}
  </part>
</score-partwise>"""


def generate_level4(seed: int, n_measures: int = 16) -> str:
    """Single staff, various keys, accidentals, varied rhythms, rests, ties.

    Builds on Level 3 (key sigs + accidentals) and adds rests and ties.
    Tests: can the model handle rests and tied notes on top of accidentals?
    """
    rng = random.Random(seed)
    divisions = 2
    fifths = rng.randint(-4, 4)

    # Same key/pitch setup as Level 3
    SHARP_ORDER = "FCGDAEB"
    FLAT_ORDER = "BEADGCF"
    altered = {}
    if fifths > 0:
        for i in range(fifths):
            altered[SHARP_ORDER[i]] = 1
    elif fifths < 0:
        for i in range(-fifths):
            altered[FLAT_ORDER[i]] = -1

    base_steps = "CDEFGAB"
    pitches_with_alter = []
    for octave in [3, 4, 5]:
        for step in base_steps:
            alter = altered.get(step, 0)
            pitches_with_alter.append((step, octave, alter))
    pitches_with_alter = [(s, o, a) for s, o, a in pitches_with_alter
                          if (o == 3 and s in "AB") or o == 4 or (o == 5 and s in "CDEFGA")]

    # Same melody generation as Level 3
    n_notes = n_measures * 6
    melody_indices = []
    current = len(pitches_with_alter) // 2
    for _ in range(n_notes):
        melody_indices.append(current)
        mid = len(pitches_with_alter) // 2
        dist = current - mid
        bias = -1 if dist > len(pitches_with_alter) // 4 else (1 if dist < -len(pitches_with_alter) // 4 else rng.choice([-1, 1]))
        r = rng.random()
        mag = 1 if r < 0.5 else (2 if r < 0.8 else rng.choice([3, 4]))
        current = max(0, min(len(pitches_with_alter) - 1, current + bias * mag))
    melody = [pitches_with_alter[i] for i in melody_indices]
    pitch_idx = 0

    # Patterns with rests (Level 3 patterns + rest patterns)
    PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        [(2, "quarter", False)] * 2 + [(1, "eighth", False)] * 4,
        [(3, "quarter", True), (1, "eighth", False), (4, "half", False)],
        [(4, "half", False), (3, "quarter", True), (1, "eighth", False)],
        [(1, "eighth", False)] * 8,
        [(6, "half", True), (2, "quarter", False)],
        # Patterns with rests
        [("R", 2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(2, "quarter", False), ("R", 2, "quarter", False), (2, "quarter", False), (2, "quarter", False)],
        [("R", 4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), ("R", 4, "half", False)],
        [("R", 8, "whole", False)],
    ]

    measures = []
    prev_pitch = None
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PATTERNS)
        notes = []
        for item in pattern:
            if isinstance(item[0], str) and item[0] == "R":
                _, dur, ntype, dotted = item
                dot_xml = "\n        <dot />" if dotted else ""
                if ntype == "whole":
                    notes.append(f"""      <note>
        <rest measure="yes" />
        <duration>{dur}</duration>
      </note>""")
                else:
                    notes.append(f"""      <note>
        <rest />
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
      </note>""")
                prev_pitch = None
            else:
                dur, ntype, dotted = item
                step, octave, alter = melody[pitch_idx % len(melody)]
                pitch_idx += 1

                # Same accidental logic as Level 3 (10% chromatic)
                acc_xml = ""
                alter_xml = ""
                if rng.random() < 0.1:
                    if alter == 0:
                        chromatic = rng.choice([1, -1])
                        alter_xml = f"\n          <alter>{chromatic}</alter>"
                        acc_xml = f"\n        <accidental>{'sharp' if chromatic == 1 else 'flat'}</accidental>"
                    else:
                        alter_xml = "\n          <alter>0</alter>"
                        acc_xml = "\n        <accidental>natural</accidental>"
                elif alter != 0:
                    alter_xml = f"\n          <alter>{alter}</alter>"

                # Tie logic
                tie_xml = ""
                if prev_pitch == (step, octave) and rng.random() < 0.3:
                    tie_xml = '\n        <tie type="stop" />'
                elif rng.random() < 0.1:
                    tie_xml = '\n        <tie type="start" />'

                midi_approx = octave * 7 + "CDEFGAB".index(step)
                stem = "up" if midi_approx < 4 * 7 + 6 else "down"
                dot_xml = "\n        <dot />" if dotted else ""

                notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}{acc_xml}{tie_xml}
        <stem>{stem}</stem>
      </note>""")
                prev_pitch = (step, octave)

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""

        measure_xml = f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>"""
        measures.append(measure_xml)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Treble</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(measures)}
  </part>
</score-partwise>"""


def _make_key_and_pitches(rng, clef="treble"):
    """Generate key signature and pitch pool for a given clef."""
    fifths = rng.randint(-4, 4)
    SHARP_ORDER = "FCGDAEB"
    FLAT_ORDER = "BEADGCF"
    altered = {}
    if fifths > 0:
        for i in range(fifths):
            altered[SHARP_ORDER[i]] = 1
    elif fifths < 0:
        for i in range(-fifths):
            altered[FLAT_ORDER[i]] = -1

    base_steps = "CDEFGAB"
    pitches = []
    if clef == "treble":
        for octave in [3, 4, 5]:
            for step in base_steps:
                pitches.append((step, octave, altered.get(step, 0)))
        pitches = [(s, o, a) for s, o, a in pitches
                   if (o == 3 and s in "AB") or o == 4 or (o == 5 and s in "CDEFGA")]
    else:  # bass
        for octave in [2, 3, 4]:
            for step in base_steps:
                pitches.append((step, octave, altered.get(step, 0)))
        pitches = [(s, o, a) for s, o, a in pitches
                   if (o == 2 and s in "CDEFGAB") or o == 3 or (o == 4 and s in "C")]

    return fifths, altered, pitches


def _generate_part_measures(rng, melody, n_measures, divisions, fifths, clef, include_rests=True):
    """Generate MusicXML measures for one part."""
    PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        [(3, "quarter", True), (1, "eighth", False), (4, "half", False)],
        [(4, "half", False), (3, "quarter", True), (1, "eighth", False)],
        [(1, "eighth", False)] * 8,
        [(6, "half", True), (2, "quarter", False)],
    ]
    if include_rests:
        PATTERNS += [
            [("R", 2, "quarter", False), (2, "quarter", False), (4, "half", False)],
            [(2, "quarter", False), ("R", 2, "quarter", False), (2, "quarter", False), (2, "quarter", False)],
            [("R", 4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        ]

    clef_sign = "G" if clef == "treble" else "F"
    clef_line = "2" if clef == "treble" else "4"
    threshold = _pitch_to_index("B4") if clef == "treble" else _pitch_to_index("D3")

    measures = []
    pitch_idx = 0
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PATTERNS)
        notes = []
        for item in pattern:
            if isinstance(item[0], str) and item[0] == "R":
                _, dur, ntype, dotted = item
                dot_xml = "\n        <dot />" if dotted else ""
                notes.append(f"""      <note>
        <rest />
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
      </note>""")
            else:
                dur, ntype, dotted = item
                step, octave, alter = melody[pitch_idx % len(melody)]
                pitch_idx += 1

                acc_xml = ""
                alter_xml = ""
                if rng.random() < 0.1:
                    if alter == 0:
                        chromatic = rng.choice([1, -1])
                        alter_xml = f"\n          <alter>{chromatic}</alter>"
                        acc_xml = f"\n        <accidental>{'sharp' if chromatic == 1 else 'flat'}</accidental>"
                    else:
                        alter_xml = "\n          <alter>0</alter>"
                        acc_xml = "\n        <accidental>natural</accidental>"
                elif alter != 0:
                    alter_xml = f"\n          <alter>{alter}</alter>"

                midi_approx = _pitch_to_index(f"{step}{octave}")
                stem = "up" if midi_approx < threshold else "down"
                dot_xml = "\n        <dot />" if dotted else ""

                notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}{acc_xml}
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>{clef_sign}</sign><line>{clef_line}</line></clef>
      </attributes>"""

        measures.append(f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>""")

    return measures


def generate_level5(seed: int, n_measures: int = 16) -> str:
    """Piano grand staff (treble + bass), monophonic per hand, accidentals, rests.

    Tests: can the model handle two staves with different clefs?
    """
    rng = random.Random(seed)
    divisions = 2
    fifths, altered, treble_pitches = _make_key_and_pitches(rng, "treble")
    _, _, bass_pitches = _make_key_and_pitches(random.Random(seed), "bass")
    # Use same fifths for bass
    bass_pitches_corrected = []
    for s, o, _ in bass_pitches:
        bass_pitches_corrected.append((s, o, altered.get(s, 0)))
    bass_pitches = bass_pitches_corrected

    n_notes = n_measures * 6
    # Generate melodies for each hand
    treble_melody_idx = []
    current = len(treble_pitches) // 2
    for _ in range(n_notes):
        treble_melody_idx.append(current)
        mid = len(treble_pitches) // 2
        dist = current - mid
        bias = -1 if dist > len(treble_pitches) // 4 else (1 if dist < -len(treble_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else rng.choice([3, 4]))
        current = max(0, min(len(treble_pitches) - 1, current + bias * mag))
    treble_melody = [treble_pitches[i] for i in treble_melody_idx]

    bass_melody_idx = []
    current = len(bass_pitches) // 2
    for _ in range(n_notes):
        bass_melody_idx.append(current)
        mid = len(bass_pitches) // 2
        dist = current - mid
        bias = -1 if dist > len(bass_pitches) // 4 else (1 if dist < -len(bass_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else rng.choice([3, 4]))
        current = max(0, min(len(bass_pitches) - 1, current + bias * mag))
    bass_melody = [bass_pitches[i] for i in bass_melody_idx]

    treble_measures = _generate_part_measures(rng, treble_melody, n_measures, divisions, fifths, "treble")
    bass_measures = _generate_part_measures(rng, bass_melody, n_measures, divisions, fifths, "bass")

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Right Hand</part-name>
    </score-part>
    <score-part id="P2">
      <part-name print-object="no">Left Hand</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(treble_measures)}
  </part>
  <part id="P2">
{chr(10).join(bass_measures)}
  </part>
</score-partwise>"""


def generate_level6(seed: int, n_measures: int = 16) -> str:
    """Piano grand staff with chords (2-3 notes per beat in right hand).

    Tests: can the model read simultaneous notes (chords)?
    """
    rng = random.Random(seed)
    divisions = 2
    fifths, altered, treble_pitches = _make_key_and_pitches(rng, "treble")
    _, _, bass_pitches = _make_key_and_pitches(random.Random(seed), "bass")
    bass_pitches = [(s, o, altered.get(s, 0)) for s, o, _ in bass_pitches]

    n_notes = n_measures * 6
    # Bass melody (monophonic)
    bass_melody_idx = []
    current = len(bass_pitches) // 2
    for _ in range(n_notes):
        bass_melody_idx.append(current)
        mid = len(bass_pitches) // 2
        dist = current - mid
        bias = -1 if dist > len(bass_pitches) // 4 else (1 if dist < -len(bass_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else rng.choice([3, 4]))
        current = max(0, min(len(bass_pitches) - 1, current + bias * mag))
    bass_melody = [bass_pitches[i] for i in bass_melody_idx]

    # Treble: generate chords (root + third + optional fifth)
    treble_melody_idx = []
    current = len(treble_pitches) // 3  # start lower for room to add chord tones
    for _ in range(n_notes):
        treble_melody_idx.append(current)
        mid = len(treble_pitches) // 3
        dist = current - mid
        bias = -1 if dist > len(treble_pitches) // 4 else (1 if dist < -len(treble_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else rng.choice([3, 4]))
        current = max(0, min(len(treble_pitches) - 4, current + bias * mag))

    # Build treble measures with chords
    PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        [(6, "half", True), (2, "quarter", False)],
    ]

    treble_measures = []
    pitch_idx = 0
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PATTERNS)
        notes = []
        for dur, ntype, dotted in pattern:
            root_idx = treble_melody_idx[pitch_idx % len(treble_melody_idx)]
            pitch_idx += 1
            # Root note
            step, octave, alter = treble_pitches[root_idx]
            alter_xml = f"\n          <alter>{alter}</alter>" if alter != 0 else ""
            dot_xml = "\n        <dot />" if dotted else ""
            midi_approx = _pitch_to_index(f"{step}{octave}")
            stem = "up" if midi_approx < _pitch_to_index("B4") else "down"
            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>
      </note>""")
            # Add chord tones (third, sometimes fifth)
            for offset in [2, 4] if rng.random() < 0.5 else [2]:
                chord_idx = min(root_idx + offset, len(treble_pitches) - 1)
                cs, co, ca = treble_pitches[chord_idx]
                ca_xml = f"\n          <alter>{ca}</alter>" if ca != 0 else ""
                notes.append(f"""      <note>
        <chord />
        <pitch>
          <step>{cs}</step>{ca_xml}
          <octave>{co}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""
        treble_measures.append(f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>""")

    bass_measures = _generate_part_measures(rng, bass_melody, n_measures, divisions, fifths, "bass")

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name print-object="no">Right Hand</part-name>
    </score-part>
    <score-part id="P2">
      <part-name print-object="no">Left Hand</part-name>
    </score-part>
  </part-list>
  <part id="P1">
{chr(10).join(treble_measures)}
  </part>
  <part id="P2">
{chr(10).join(bass_measures)}
  </part>
</score-partwise>"""


def generate_level7_parameterized(
    seed: int,
    n_measures: int = 24,
    include_voice: bool = True,
    include_lyrics: bool = True,
) -> str:
    """Parameterized version of Level 7 for ablation studies.

    - include_voice=True, include_lyrics=True, n_measures=24: standard Level 7
    - include_voice=True, include_lyrics=False: Level 7 without lyrics (7a/7c)
    - include_voice=False: piano-only grand staff with chords (6b)
    """
    rng = random.Random(seed)
    divisions = 2
    fifths, altered, treble_pitches = _make_key_and_pitches(rng, "treble")
    _, _, bass_pitches = _make_key_and_pitches(random.Random(seed), "bass")
    bass_pitches = [(s, o, altered.get(s, 0)) for s, o, _ in bass_pitches]

    # Voice melody (treble range, simpler rhythms) — only generated if include_voice
    voice_measures = []
    VOICE_PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(3, "quarter", True), (1, "eighth", False), (4, "half", False)],
    ]

    # Simple English syllables for lyrics
    SYLLABLES = [
        ("The", "single"), ("sun", "single"), ("will", "single"), ("shine", "single"),
        ("to", "begin"), ("day", "end"), ("through", "single"), ("the", "single"),
        ("morn", "begin"), ("ing", "end"), ("light", "single"), ("and", "single"),
        ("eve", "begin"), ("ning", "end"), ("stars", "single"), ("a", "begin"),
        ("bove", "end"), ("sing", "single"), ("a", "single"), ("song", "single"),
        ("of", "single"), ("love", "single"), ("and", "single"), ("peace", "single"),
        ("for", "single"), ("all", "single"), ("who", "single"), ("hear", "single"),
        ("the", "single"), ("mu", "begin"), ("sic", "end"), ("play", "single"),
    ]

    if include_voice:
        n_notes = n_measures * 4  # voice is simpler
        voice_melody_idx = []
        current = len(treble_pitches) // 2
        for _ in range(n_notes):
            voice_melody_idx.append(current)
            mid = len(treble_pitches) // 2
            dist = current - mid
            bias = -1 if dist > len(treble_pitches) // 4 else (1 if dist < -len(treble_pitches) // 4 else rng.choice([-1, 1]))
            mag = 1 if rng.random() < 0.5 else 2
            current = max(0, min(len(treble_pitches) - 1, current + bias * mag))

        pitch_idx = 0
        syl_idx = 0
    else:
        # Still consume rng to keep melody reproducible across variants
        n_notes = n_measures * 4
        _ = [rng.random() for _ in range(n_notes * 3)]
        voice_melody_idx = []
        pitch_idx = 0
        syl_idx = 0

    for m in range(1, n_measures + 1) if include_voice else range(0):
        pattern = rng.choice(VOICE_PATTERNS)
        notes = []
        for dur, ntype, dotted in pattern:
            step, octave, alter = treble_pitches[voice_melody_idx[pitch_idx % len(voice_melody_idx)]]
            pitch_idx += 1
            alter_xml = f"\n          <alter>{alter}</alter>" if alter != 0 else ""
            dot_xml = "\n        <dot />" if dotted else ""
            midi_approx = _pitch_to_index(f"{step}{octave}")
            stem = "up" if midi_approx < _pitch_to_index("B4") else "down"

            if include_lyrics:
                text, syllabic = SYLLABLES[syl_idx % len(SYLLABLES)]
                syl_idx += 1
                lyric_xml = f"""
        <lyric number="1">
          <syllabic>{syllabic}</syllabic>
          <text>{text}</text>
        </lyric>"""
            else:
                lyric_xml = ""

            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>{lyric_xml}
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""
        voice_measures.append(f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>""")

    # Piano RH: chords (like Level 6)
    n_piano_notes = n_measures * 6
    PIANO_PATTERNS = [
        [(2, "quarter", False)] * 4,
        [(4, "half", False), (2, "quarter", False), (2, "quarter", False)],
        [(2, "quarter", False), (2, "quarter", False), (4, "half", False)],
        [(1, "eighth", False)] * 4 + [(2, "quarter", False)] * 2,
        [(6, "half", True), (2, "quarter", False)],
    ]
    treble_mel_idx = []
    current = len(treble_pitches) // 3
    for _ in range(n_piano_notes):
        treble_mel_idx.append(current)
        mid = len(treble_pitches) // 3
        dist = current - mid
        bias = -1 if dist > len(treble_pitches) // 4 else (1 if dist < -len(treble_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else 3)
        current = max(0, min(len(treble_pitches) - 4, current + bias * mag))

    piano_treble_measures = []
    p_idx = 0
    for m in range(1, n_measures + 1):
        pattern = rng.choice(PIANO_PATTERNS)
        notes = []
        for dur, ntype, dotted in pattern:
            root_idx = treble_mel_idx[p_idx % len(treble_mel_idx)]
            p_idx += 1
            step, octave, alter = treble_pitches[root_idx]
            alter_xml = f"\n          <alter>{alter}</alter>" if alter != 0 else ""
            dot_xml = "\n        <dot />" if dotted else ""
            midi_approx = _pitch_to_index(f"{step}{octave}")
            stem = "up" if midi_approx < _pitch_to_index("B4") else "down"
            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>{alter_xml}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>
      </note>""")
            # Add chord tones
            for offset in [2, 4] if rng.random() < 0.5 else [2]:
                chord_idx = min(root_idx + offset, len(treble_pitches) - 1)
                cs, co, ca = treble_pitches[chord_idx]
                ca_xml = f"\n          <alter>{ca}</alter>" if ca != 0 else ""
                notes.append(f"""      <note>
        <chord />
        <pitch>
          <step>{cs}</step>{ca_xml}
          <octave>{co}</octave>
        </pitch>
        <duration>{dur}</duration>
        <type>{ntype}</type>{dot_xml}
        <stem>{stem}</stem>
      </note>""")

        attrs = ""
        if m == 1:
            attrs = f"""      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>"""
        piano_treble_measures.append(f"""    <measure number="{m}">
{attrs}
{chr(10).join(notes)}
    </measure>""")

    # Piano LH: monophonic bass (with rests)
    bass_mel = []
    current = len(bass_pitches) // 2
    for _ in range(n_piano_notes):
        bass_mel.append(bass_pitches[current])
        mid = len(bass_pitches) // 2
        dist = current - mid
        bias = -1 if dist > len(bass_pitches) // 4 else (1 if dist < -len(bass_pitches) // 4 else rng.choice([-1, 1]))
        mag = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.8 else 3)
        current = max(0, min(len(bass_pitches) - 1, current + bias * mag))

    piano_bass = _generate_part_measures(rng, bass_mel, n_measures, divisions, fifths, "bass")

    if include_voice:
        part_list = """    <score-part id="P1">
      <part-name print-object="no">Voice</part-name>
    </score-part>
    <score-part id="P2">
      <part-name print-object="no">Piano RH</part-name>
    </score-part>
    <score-part id="P3">
      <part-name print-object="no">Piano LH</part-name>
    </score-part>"""
        parts_xml = f"""  <part id="P1">
{chr(10).join(voice_measures)}
  </part>
  <part id="P2">
{chr(10).join(piano_treble_measures)}
  </part>
  <part id="P3">
{chr(10).join(piano_bass)}
  </part>"""
    else:
        part_list = """    <score-part id="P1">
      <part-name print-object="no">Piano RH</part-name>
    </score-part>
    <score-part id="P2">
      <part-name print-object="no">Piano LH</part-name>
    </score-part>"""
        # Renumber piano parts from P2/P3 to P1/P2
        piano_treble_p1 = [m.replace('', '') for m in piano_treble_measures]  # no-op, part id is at wrapper level
        parts_xml = f"""  <part id="P1">
{chr(10).join(piano_treble_measures)}
  </part>
  <part id="P2">
{chr(10).join(piano_bass)}
  </part>"""

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
{part_list}
  </part-list>
{parts_xml}
</score-partwise>"""


def generate_level7(seed: int, n_measures: int = 24) -> str:
    """Standard Level 7: voice+piano+chords+lyrics, 24 measures."""
    return generate_level7_parameterized(seed, n_measures=24, include_voice=True, include_lyrics=True)


def generate_level6b(seed: int) -> str:
    """Level 6b: Piano-only grand staff with chords, 24 measures.
    Uses Level 6's generator with more measures (isolates length from Level 6).
    """
    return generate_level6(seed, n_measures=24)


def generate_level7a(seed: int) -> str:
    """Level 7a: Voice+piano, 16 measures, NO lyrics (isolates voice part addition)."""
    return generate_level7_parameterized(seed, n_measures=16, include_voice=True, include_lyrics=False)


def generate_level7b(seed: int) -> str:
    """Level 7b: Voice+piano, 16 measures, WITH lyrics (isolates lyrics markup)."""
    return generate_level7_parameterized(seed, n_measures=16, include_voice=True, include_lyrics=True)


def generate_level7c(seed: int) -> str:
    """Level 7c: Voice+piano, 24 measures, NO lyrics (isolates length from 7a)."""
    return generate_level7_parameterized(seed, n_measures=24, include_voice=True, include_lyrics=False)


GENERATORS = {
    1: generate_level1,
    2: generate_level2,
    3: generate_level3,
    4: generate_level4,
    5: generate_level5,
    6: generate_level6,
    7: generate_level7,
    "6b": generate_level6b,
    "7a": generate_level7a,
    "7b": generate_level7b,
    "7c": generate_level7c,
}


def render_with_lilypond(musicxml_path: str, output_png: str) -> bool:
    """Render MusicXML to PNG using the cvlization/lilypond Docker image."""
    musicxml_path = os.path.abspath(musicxml_path)
    output_png = os.path.abspath(output_png)
    output_dir = os.path.dirname(output_png)
    basename = os.path.splitext(os.path.basename(output_png))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy MusicXML to temp dir
        tmp_mxml = os.path.join(tmpdir, "score.musicxml")
        with open(tmp_mxml, "w") as f:
            f.write(open(musicxml_path).read())

        # LilyPond layout overrides: match openscore page dimensions,
        # remove tagline, use compact layout
        ly_overrides = r"""
\paper {
  #(set-paper-size "a4")
  tagline = ##f
  indent = 0
  top-margin = 10
  bottom-margin = 10
  left-margin = 15
  right-margin = 15
  ragged-bottom = ##t
  ragged-last-bottom = ##t
}
\header { title = ##f tagline = ##f }
\layout {
  \context {
    \Staff
    instrumentName = ##f
    shortInstrumentName = ##f
    \remove "Bar_number_engraver"
  }
}
"""
        override_path = os.path.join(tmpdir, "overrides.ly")
        with open(override_path, "w") as f:
            f.write(ly_overrides)

        # Run LilyPond via Docker: convert MusicXML → .ly, inject overrides, render
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/data",
            "cvlization/lilypond:latest",
            "bash", "-c",
            "cd /data && musicxml2ly score.musicxml -o score.ly 2>/dev/null && "
            "sed -i 's/instrumentName = .*/instrumentName = ##f/' score.ly && "
            "sed -i 's/shortInstrumentName = .*/shortInstrumentName = ##f/' score.ly && "
            "cat overrides.ly >> score.ly && "
            "lilypond --png -dresolution=150 score.ly 2>/dev/null"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                print(f"  LilyPond error: {result.stderr.decode()[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  LilyPond timeout")
            return False

        # Find output PNG (LilyPond names it score.png or score-page1.png)
        for candidate in ["score.png", "score-page1.png"]:
            src = os.path.join(tmpdir, candidate)
            if os.path.exists(src):
                os.makedirs(output_dir, exist_ok=True)
                # Crop whitespace: trim bottom, keep some margin
                try:
                    from PIL import Image as PILImage, ImageOps
                    img = PILImage.open(src)
                    # Find bounding box of non-white content
                    gray = img.convert("L")
                    bbox = ImageOps.invert(gray).getbbox()
                    if bbox:
                        # Add margin around content
                        margin = 40
                        crop_box = (0, 0, img.width, min(img.height, bbox[3] + margin))
                        img = img.crop(crop_box)
                    img.save(output_png)
                except ImportError:
                    subprocess.run(["cp", src, output_png])
                return True

        print(f"  No PNG output found in {tmpdir}")
        return False


def batch_render(out_dir, filenames):
    """Render all MusicXML files in a single Docker container."""
    out_dir = Path(out_dir).resolve()

    # Write the LilyPond overrides once
    ly_overrides = r"""\paper {
  #(set-paper-size "a4")
  tagline = ##f
  indent = 0
  top-margin = 10
  bottom-margin = 10
  left-margin = 15
  right-margin = 15
  ragged-bottom = ##t
  ragged-last-bottom = ##t
}
\header { title = ##f tagline = ##f }
\layout {
  \context {
    \Score
    \override BarNumber.break-visibility = ##(#f #f #f)
  }
}
"""
    (out_dir / "_overrides.ly").write_text(ly_overrides)

    # Build a bash script that converts + renders all files
    lines = ["#!/bin/bash", "cd /data"]
    for name in filenames:
        lines.append(
            f"musicxml2ly {name}.musicxml -o {name}.ly 2>/dev/null && "
            f"sed -i 's/instrumentName = .*/instrumentName = ##f/' {name}.ly && "
            f"sed -i 's/shortInstrumentName = .*/shortInstrumentName = ##f/' {name}.ly && "
            r"sed -i 's/\\context { \\Score/\\context { \\Score \\override BarNumber.break-visibility = ##(#f #f #f)/' " + f"{name}.ly && "
            f"cat _overrides.ly >> {name}.ly && "
            f"lilypond --png -dresolution=150 {name}.ly 2>/dev/null && "
            f"echo OK:{name} || echo FAIL:{name}"
        )
    script = "\n".join(lines)
    (out_dir / "_render.sh").write_text(script)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{out_dir}:/data",
        "cvlization/lilypond:latest",
        "bash", "/data/_render.sh",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=3600)
    output = result.stdout.decode()

    # Crop rendered PNGs
    rendered = 0
    try:
        from PIL import Image as PILImage, ImageOps
        has_pil = True
    except ImportError:
        has_pil = False

    for name in filenames:
        # LilyPond outputs name.png or name-page1.png
        for candidate in [out_dir / f"{name}.png", out_dir / f"{name}-page1.png"]:
            if candidate.exists():
                if has_pil:
                    img = PILImage.open(candidate)
                    gray = img.convert("L")
                    bbox = ImageOps.invert(gray).getbbox()
                    if bbox:
                        margin = 40
                        img = img.crop((0, 0, img.width, min(img.height, bbox[3] + margin)))
                    img.save(out_dir / f"{name}.png")
                    # Remove -page1 variant if different
                    if candidate.name != f"{name}.png" and candidate.exists():
                        candidate.unlink()
                rendered += 1
                break

    # Cleanup temp files
    for f in out_dir.glob("*.ly"):
        f.unlink()
    (out_dir / "_overrides.ly").unlink(missing_ok=True)
    (out_dir / "_render.sh").unlink(missing_ok=True)

    if rendered % 100 == 0 or rendered == len(filenames):
        print(f"  Rendered {rendered}/{len(filenames)}")

    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    def _level_type(v):
        try:
            return int(v)
        except ValueError:
            return v
    parser.add_argument("--level", type=_level_type, default=1, choices=list(GENERATORS.keys()),
                        help="Difficulty level (default: 1). Supports 1-7 and variants 6b/7a/7b/7c.")
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of scores to generate (default: 1000)")
    parser.add_argument("--measures", type=int, default=16,
                        help="Measures per score (default: 16)")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--render", action="store_true",
                        help="Render to PNG with LilyPond Docker")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed (default: 0)")
    args = parser.parse_args()

    generator = GENERATORS[args.level]
    out_dir = Path(args.output) / f"level{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Variant generators (6b, 7a, 7b, 7c) have hardcoded n_measures — ignore CLI --measures
    is_variant = isinstance(args.level, str)
    print(f"Generating {args.count} level-{args.level} scores...")

    # Generate all MusicXML files
    filenames = []
    for i in range(args.count):
        seed = args.seed_start + i
        if is_variant:
            musicxml = generator(seed)
        else:
            musicxml = generator(seed, n_measures=args.measures)
        name = f"L{args.level}_{seed:05d}"
        mxml_path = out_dir / f"{name}.musicxml"
        mxml_path.write_text(musicxml)
        filenames.append(name)

    print(f"  Generated {len(filenames)} MusicXML files")

    if args.render:
        print("Rendering with LilyPond (batch)...")
        rendered = batch_render(out_dir, filenames)
        print(f"  Rendered: {rendered}/{len(filenames)}")

    print(f"\nDone. {args.count} scores in {out_dir}")


if __name__ == "__main__":
    main()
