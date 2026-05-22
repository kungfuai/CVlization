"""Build a stratified KD-dataset manifest from Hallo-Live's prompt CSV.

Each Hallo prompt is structured as:
    <scene description> <S>spoken line<E> <S>...<E> <AUDCAP>...<ENDAUDCAP>

We parse each into:
    scene        - text before the first <S> tag (avatar appearance / setting)
    spoken_text  - concatenation of all <S>...<E> contents (what CosyVoice says)
    full_prompt  - the original string (text conditioning for SoulX / the student)

Prompts with no <S> tag are skipped (no speech -> nothing for TTS).

Stratified sampling by visual style so the dataset isn't all realistic humans:
    human  ~70%   default
    anime  ~20%   keyword: anime|cartoon|toon|stylized|illustrated|hand-drawn
    edge   ~10%   keyword: dwarf|cyclops|creature|goblin|elf|orc|fantasy|robot|alien

Output: JSONL, one item per line:
    {"id": "00007", "bucket": "human", "scene": "...", "spoken_text": "...",
     "full_prompt": "..."}
"""
import argparse
import csv
import json
import random
import re

ANIME_RE = re.compile(r"\b(anime|cartoon|toon|stylized|illustrated|hand-drawn|3d animation)\b", re.I)
EDGE_RE = re.compile(r"\b(dwarf|cyclops|creature|goblin|elf|orc|fantasy|robot|alien|monster|mermaid)\b", re.I)
SPOKEN_RE = re.compile(r"<S>(.*?)<E>", re.S)


def parse_prompt(text):
    """Return (scene, spoken_text) or (scene, None) if no spoken lines."""
    spoken = [s.strip() for s in SPOKEN_RE.findall(text)]
    spoken_text = " ".join(spoken).strip() if spoken else None
    # scene = everything before the first <S>; fall back to whole text
    scene = re.split(r"<S>", text, maxsplit=1)[0].strip()
    # strip any stray <AUDCAP>... that appears before <S> (rare)
    scene = re.split(r"<AUDCAP>", scene, maxsplit=1)[0].strip()
    return scene, spoken_text


def bucket_of(text):
    if ANIME_RE.search(text):
        return "anime"
    if EDGE_RE.search(text):
        return "edge"
    return "human"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Hallo synthetic_prompts_32k.csv")
    ap.add_argument("--out", required=True, help="output manifest.jsonl")
    ap.add_argument("--n", type=int, default=100, help="total items")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio-human", type=float, default=0.70)
    ap.add_argument("--ratio-anime", type=float, default=0.20)
    ap.add_argument("--ratio-edge", type=float, default=0.10)
    args = ap.parse_args()

    random.seed(args.seed)

    # Bucket every usable prompt
    buckets = {"human": [], "anime": [], "edge": []}
    n_total, n_no_speech = 0, 0
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            text = row["text_prompt"]
            scene, spoken = parse_prompt(text)
            if not spoken or not scene:
                n_no_speech += 1
                continue
            buckets[bucket_of(text)].append(
                {"scene": scene, "spoken_text": spoken, "full_prompt": text}
            )

    print(f"Parsed {n_total} prompts ({n_no_speech} skipped: no <S> speech).")
    for b, items in buckets.items():
        print(f"  {b}: {len(items)} candidates")

    targets = {
        "human": round(args.n * args.ratio_human),
        "anime": round(args.n * args.ratio_anime),
        "edge": round(args.n * args.ratio_edge),
    }
    # fix rounding drift
    drift = args.n - sum(targets.values())
    targets["human"] += drift

    selected = []
    for b, want in targets.items():
        pool = buckets[b]
        if len(pool) < want:
            print(f"  WARNING: {b} has only {len(pool)} < {want} requested; taking all")
            want = len(pool)
        selected.extend((b, it) for it in random.sample(pool, want))

    random.shuffle(selected)
    with open(args.out, "w") as f:
        for i, (b, it) in enumerate(selected):
            rec = {"id": f"{i:05d}", "bucket": b, **it}
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(selected)} items to {args.out}")
    final = {}
    for b, _ in selected:
        final[b] = final.get(b, 0) + 1
    print(f"  final mix: {final}")


if __name__ == "__main__":
    main()
