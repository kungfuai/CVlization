# Sample outputs (deterministic mock-brain run)

These are the verified artifacts from the default **`MockBrain`** run — no API
key, no network, fully deterministic — so reviewers can inspect the pipeline's
output without running anything. Regenerate them exactly with:

```bash
M=$(mktemp -d)
python generate.py --memory-dir "$M" --output verified_run/deck.json
python revise.py   --memory-dir "$M" --deck verified_run/deck.json \
    --feedback-file data/feedback.example.txt \
    --output verified_run/deck_revised.json --report verified_run/revision_report.json
python evaluate.py --memory-dir "$M/eval" --fresh --report verified_run/evaluate_report.json
```

| File | What it shows |
|------|---------------|
| [`deck.txt`](deck.txt) / [`deck.json`](deck.json) | Round-0 deck, personalized from the long-term profile (executive tone → "Bottom line:" prefix, 4-bullet cap, light theme, `licensing` avoided). |
| [`deck_revised.txt`](deck_revised.txt) / [`deck_revised.json`](deck_revised.json) | After 4 scoped local-revision turns: slide 3 & 4 shortened, theme flipped to **dark**, a **Pricing** slide inserted — unrelated slides untouched. |
| [`revision_report.json`](revision_report.json) | Per-turn edit plan, edit surface (`slides_touched`, `preservation_ratio`), tool-recipe reuse, working-memory constraints, and the profile after the durable "always dark theme" promotion. |
| [`evaluate_report.json`](evaluate_report.json) | The three-check comparison: scoped-vs-full, host-cache memory reuse, and preference adherence. |

> Provider coverage: these are mock-brain outputs by design. The real-provider
> path (`openai`/`groq`/`ollama` via LiteLLM) is wired but intentionally left
> untested here (requires an API key).
