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


GENERATORS = {
    1: generate_level1,
    2: generate_level2,
    3: generate_level3,
    4: generate_level4,
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
    parser.add_argument("--level", type=int, default=1, choices=list(GENERATORS.keys()),
                        help="Difficulty level (default: 1)")
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

    print(f"Generating {args.count} level-{args.level} scores ({args.measures} measures each)...")

    # Generate all MusicXML files
    filenames = []
    for i in range(args.count):
        seed = args.seed_start + i
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
