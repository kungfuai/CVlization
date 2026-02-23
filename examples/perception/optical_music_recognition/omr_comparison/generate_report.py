#!/usr/bin/env python3
"""
Generate an HTML comparison report from OMR output JSON files.

Usage:
    python generate_report.py [--output report.html] [--image path/to/score.jpg]
"""

import argparse
import base64
import json
import re
from pathlib import Path

PROMPT_ORDER = [
    "key_signature",
    "time_signature",
    "musical_era",
    "dynamics_tempo",
    "ekern_transcription",
]

PROMPT_LABELS = {
    "key_signature": "Key Signature",
    "time_signature": "Time Signature",
    "musical_era": "Musical Era",
    "dynamics_tempo": "Dynamics & Tempo",
    "ekern_transcription": "Ekern Transcription",
}

# Ground truth for color-coding summary
GROUND_TRUTH_NOTES = {
    "key_signature": "~2 sharps (D major) in Intro, key change later",
    "time_signature": "3/4",
    "musical_era": "Romantic / late 19th century (1884)",
    "dynamics_tempo": "pp, ben marcato, cres., f, p, marcato, glissando, legato p, fz",
    "ekern_transcription": "(no reliable ground truth — structural validity only)",
}

MODEL_DISPLAY = {
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro ⭐",
    "gpt-4o": "GPT-4o",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B (local)",
    "smt": "SMT-OMR (fine-tuned)",
}

# Per-model, per-prompt manual quality annotations
ANNOTATIONS = {
    "gemini/gemini-2.5-flash": {
        "key_signature": ("warn", "G major (1♯) — wrong"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", ""),
        "ekern_transcription": ("bad", "Aborted after header"),
    },
    "gemini/gemini-3.1-pro-preview": {
        "key_signature": ("warn", "1♯ + detected key change"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", ""),
        "ekern_transcription": ("good", "Best VLM attempt — actual notes"),
    },
    "gpt-4o": {
        "key_signature": ("warn", "D major (2♯) — closest"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Late Romantic ✓"),
        "dynamics_tempo": ("good", ""),
        "ekern_transcription": ("bad", "Refused"),
    },
    "gpt-5.2": {
        "key_signature": ("bad", "C major (0) — wrong"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", "Most detailed"),
        "ekern_transcription": ("warn", "Declined gracefully"),
    },
    "gpt-5.2-pro": {
        "key_signature": ("warn", ""),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", ""),
        "ekern_transcription": ("warn", ""),
    },
    "anthropic/claude-sonnet-4-6": {
        "key_signature": ("bad", "Bb major (2♭) — wrong"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", "Most structured"),
        "ekern_transcription": ("warn", "Partial notes — hallucinated"),
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "key_signature": ("bad", "F# major (1♯) — wrong"),
        "time_signature": ("good", "3/4 ✓"),
        "musical_era": ("good", "Romantic ✓"),
        "dynamics_tempo": ("good", ""),
        "ekern_transcription": ("bad", "Header loop"),
    },
}

SMT_ANNOTATIONS = {
    "key_signature": ("warn", "*k[b-] = F major (1♭)"),
    "time_signature": ("bad", "*M2/4 — wrong (should be 3/4)"),
    "ekern_transcription": ("good", "Complete bekern output"),
}


def load_records(comparison_dir: Path) -> list[dict]:
    records = []
    for f in sorted(comparison_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "results" in data:
                data["_file"] = f.name
                records.append(data)
        except Exception:
            pass
    return records


def load_smt(comparison_dir: Path) -> dict | None:
    p = comparison_dir / "smt_omr.json"
    if not p.exists():
        return None
    return {"_file": "smt_omr.json", "model": "smt", "ekern": p.read_text()}


def encode_image(image_path: Path) -> str | None:
    if not image_path or not image_path.exists():
        return None
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def md_to_html(text: str) -> str:
    """Minimal markdown → HTML for bold and newlines."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = text.replace("\n", "<br>")
    return text


def badge(level: str, note: str) -> str:
    colors = {"good": "#2d7a2d", "warn": "#b86e00", "bad": "#a02020"}
    bg = {"good": "#e6f4ea", "warn": "#fff3cd", "bad": "#fce8e8"}
    if not note:
        return ""
    return (
        f'<span style="background:{bg.get(level,"#eee")};color:{colors.get(level,"#333")};'
        f'padding:2px 6px;border-radius:3px;font-size:0.82em;white-space:nowrap">{note}</span>'
    )


def render_html(records: list[dict], smt: dict | None, image_b64: str | None) -> str:
    model_ids = [r["model"] for r in records]

    # Summary table rows
    summary_rows = ""
    for pid in PROMPT_ORDER:
        label = PROMPT_LABELS[pid]
        gt = GROUND_TRUTH_NOTES[pid]
        cells = f"<td><strong>{label}</strong><br><small style='color:#666'>{gt}</small></td>"
        for r in records:
            mid = r["model"]
            ann = ANNOTATIONS.get(mid, {}).get(pid, ("", ""))
            lvl, note = ann if ann else ("", "")
            bg = {"good": "#e6f4ea", "warn": "#fff3cd", "bad": "#fce8e8"}.get(lvl, "")
            resp = r.get("results", {}).get(pid, {}).get("response", "—")
            preview = resp[:120].replace("\n", " ") + ("…" if len(resp) > 120 else "")
            cells += (
                f'<td style="background:{bg};vertical-align:top;font-size:0.85em">'
                f"{badge(lvl, note)}<br><span style='color:#444'>{preview}</span></td>"
            )
        if smt:
            if pid == "ekern_transcription":
                lvl, note = SMT_ANNOTATIONS.get(pid, ("", ""))
                cells += (
                    f'<td style="background:#e6f4ea;vertical-align:top;font-size:0.85em">'
                    f"{badge(lvl, note)}</td>"
                )
            elif pid in SMT_ANNOTATIONS:
                lvl, note = SMT_ANNOTATIONS[pid]
                bg = {"good": "#e6f4ea", "warn": "#fff3cd", "bad": "#fce8e8"}.get(lvl, "")
                cells += f'<td style="background:{bg};vertical-align:top;font-size:0.85em">{badge(lvl, note)}</td>'
            else:
                cells += "<td style='color:#999'>N/A</td>"
        summary_rows += f"<tr>{cells}</tr>\n"

    # Header row
    header = "<th>Prompt</th>"
    for r in records:
        mid = r["model"]
        display = MODEL_DISPLAY.get(mid, mid)
        mtype = "API" if not mid.startswith("Qwen") else "Local GPU"
        header += f"<th>{display}<br><small style='color:#888'>{mtype}</small></th>"
    if smt:
        header += "<th>SMT-OMR<br><small style='color:#888'>Local GPU (fine-tuned)</small></th>"

    # Per-prompt detail sections
    detail_sections = ""
    for pid in PROMPT_ORDER:
        label = PROMPT_LABELS[pid]
        gt = GROUND_TRUTH_NOTES[pid]
        cards = ""
        for r in records:
            mid = r["model"]
            display = MODEL_DISPLAY.get(mid, mid)
            resp = r.get("results", {}).get(pid, {}).get("response", "—")
            ann = ANNOTATIONS.get(mid, {}).get(pid, ("", ""))
            lvl, note = ann if ann else ("", "")
            b = badge(lvl, note)
            is_ekern = pid == "ekern_transcription"
            content = (
                f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;"
                f"overflow-x:auto;font-size:0.78em;white-space:pre-wrap'>{resp}</pre>"
                if is_ekern
                else f"<p style='font-size:0.88em'>{md_to_html(resp)}</p>"
            )
            cards += (
                f'<div style="border:1px solid #ddd;border-radius:6px;padding:12px;'
                f'margin-bottom:12px;min-width:280px;max-width:400px;flex:1 1 280px">'
                f"<strong>{display}</strong> {b}{content}</div>\n"
            )
        if smt and pid == "ekern_transcription":
            lvl, note = SMT_ANNOTATIONS.get(pid, ("good", ""))
            b = badge(lvl, note)
            cards += (
                f'<div style="border:1px solid #ddd;border-radius:6px;padding:12px;'
                f'margin-bottom:12px;min-width:280px;max-width:400px;flex:1 1 280px">'
                f"<strong>SMT-OMR (fine-tuned)</strong> {b}"
                f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;"
                f"overflow-x:auto;font-size:0.78em;white-space:pre-wrap'>{smt['ekern']}</pre></div>"
            )
        detail_sections += (
            f'<section style="margin-bottom:40px">'
            f'<h2 style="border-bottom:2px solid #333;padding-bottom:4px">{label}</h2>'
            f'<p style="color:#555;font-size:0.9em"><strong>Ground truth:</strong> {gt}</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px">{cards}</div>'
            f"</section>\n"
        )

    img_html = (
        f'<img src="{image_b64}" style="max-height:500px;border:1px solid #ccc;border-radius:4px">'
        if image_b64
        else "<p><em>(score image not found)</em></p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OMR Model Comparison — Biddle's Piano Waltz (1884)</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; padding: 24px; color: #222; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.2em; }}
</style>
</head>
<body>
<h1>OMR Model Comparison — Biddle's Piano Waltz (1884)</h1>
<p>Zero-shot OMR outputs from all models on the same vintage score.
<strong>Run date:</strong> 2026-02-23 &nbsp;|&nbsp;
<strong>Image:</strong> Library of Congress / zzsi/cvl HuggingFace dataset</p>

<div style="display:flex;gap:32px;align-items:flex-start;margin-bottom:32px;flex-wrap:wrap">
  <div>{img_html}</div>
  <div style="flex:1;min-width:260px">
    <h2 style="margin-top:0">Score details</h2>
    <ul>
      <li><strong>Title:</strong> Biddle's Piano Waltz</li>
      <li><strong>Composer:</strong> Robert D. Biddle</li>
      <li><strong>Date:</strong> 1884</li>
      <li><strong>Source:</strong> Library of Congress American Sheet Music</li>
      <li><strong>Key:</strong> ~2 sharps (D major) intro, key change later</li>
      <li><strong>Time:</strong> 3/4</li>
      <li><strong>Era:</strong> Romantic</li>
    </ul>
    <h2>Legend</h2>
    <p>{badge("good","Correct / good")} &nbsp; {badge("warn","Partial / debatable")} &nbsp; {badge("bad","Wrong / failed")}</p>
  </div>
</div>

<h2>Summary Table</h2>
<div style="overflow-x:auto">
<table>
<thead><tr>{header}</tr></thead>
<tbody>{summary_rows}</tbody>
</table>
</div>

<h2>Full Responses by Prompt</h2>
{detail_sections}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="report.html")
    parser.add_argument("--image", default=None, help="Path to score image")
    parser.add_argument("--dir", default=".", help="Directory with JSON files")
    args = parser.parse_args()

    comparison_dir = Path(args.dir)
    records = load_records(comparison_dir)
    smt = load_smt(comparison_dir)

    image_path = Path(args.image) if args.image else None
    if not image_path:
        default = Path.home() / ".cache/huggingface/cvl_data/qwen3_omr/vintage_score_1884.jpg"
        if default.exists():
            image_path = default
    image_b64 = encode_image(image_path)

    html = render_html(records, smt, image_b64)
    out = comparison_dir / args.output
    out.write_text(html, encoding="utf-8")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
