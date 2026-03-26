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


GENERATORS = {
    1: generate_level1,
    # Future levels will be added here
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
