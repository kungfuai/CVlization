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


def generate_level1(seed: int, n_measures: int = 8) -> str:
    """Single staff, C major, quarter notes only, no rests.

    Tests: can the model read pitch from staff position at all?
    """
    rng = random.Random(seed)
    # C major scale pitches (octave 4 and 5)
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"]
    divisions = 1  # quarter note = 1 division

    measures = []
    for m in range(1, n_measures + 1):
        notes = []
        for beat in range(4):  # 4/4 time
            pitch = rng.choice(pitches)
            step = pitch[0]
            octave = pitch[-1]
            notes.append(f"""      <note>
        <pitch>
          <step>{step}</step>
          <octave>{octave}</octave>
        </pitch>
        <duration>{divisions}</duration>
        <type>quarter</type>
        <stem>{"up" if int(octave) <= 4 and step in "CDEF" else "down"}</stem>
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
  <work>
    <work-title>Synthetic Level 1 (seed {seed})</work-title>
  </work>
  <part-list>
    <score-part id="P1">
      <part-name>Treble</part-name>
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

        # Run LilyPond via Docker
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/data",
            "cvlization/lilypond:latest",
            "bash", "-c",
            "cd /data && musicxml2ly score.musicxml -o score.ly 2>/dev/null && "
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
                subprocess.run(["cp", src, output_png])
                return True

        print(f"  No PNG output found in {tmpdir}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=1, choices=list(GENERATORS.keys()),
                        help="Difficulty level (default: 1)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of scores to generate (default: 100)")
    parser.add_argument("--measures", type=int, default=8,
                        help="Measures per score (default: 8)")
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

    rendered = 0
    for i in range(args.count):
        seed = args.seed_start + i
        musicxml = generator(seed, n_measures=args.measures)

        mxml_path = out_dir / f"L{args.level}_{seed:05d}.musicxml"
        mxml_path.write_text(musicxml)

        if args.render:
            png_path = out_dir / f"L{args.level}_{seed:05d}.png"
            if render_with_lilypond(str(mxml_path), str(png_path)):
                rendered += 1
                if (rendered % 10) == 0:
                    print(f"  Rendered {rendered}/{i+1}...")
            else:
                print(f"  FAILED: {mxml_path.name}")

    print(f"\nDone. {args.count} MusicXML files in {out_dir}")
    if args.render:
        print(f"  Rendered: {rendered}/{args.count}")


if __name__ == "__main__":
    main()
