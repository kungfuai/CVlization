#!/usr/bin/env python3
"""Headless browser-use agent pointed at a local OpenAI-compatible vLLM endpoint.

Reads the task from CLI args, talks to an LLM at $OPENCODE_BASE_URL with
api_key $VLLM_API_KEY, model $BROWSER_USE_MODEL (the served-model-name
from `cvl run vllm serve`). Browser is headless Chromium; the IN_DOCKER
env (set in the Dockerfile) makes browser-use auto-disable sandboxing
inside the container.

Default task is a multi-step Wikipedia lookup that exercises navigate
+ search + click + extract. After every run the agent's history JSON,
per-step screenshots, and a markdown report all land in
$BROWSER_USE_OUTPUT_DIR (default /work/outputs, which run.sh mounts to
./outputs on the host) so the user has tangible artifacts to inspect.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "sk-local"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TASK = (
    "Open https://en.wikipedia.org. Use the search box at the top of "
    "the page to search for 'Linux'. Click the first result. From the "
    "article, find and report the year that Linux was first released. "
    "Reply with only the four-digit year on a single line, no "
    "commentary or formatting."
)
DEFAULT_OUTPUT_DIR = "/work/outputs"


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


# ---------------------------------------------------------------------------
# Optional structured-output Pydantic models. Selected at runtime via
# --output-model / $BROWSER_USE_OUTPUT_MODEL. When set, browser-use replaces
# its default DoneAction with a StructuredOutputAction(data=ModelInstance),
# and Pydantic schema validation forces the model to fill every required
# field -- bypassing the DoneAction.text "ONLY report observed data, don't
# fill gaps with training knowledge" guidance that otherwise blocks tasks
# requiring a synthesis paragraph after structured extraction.
# ---------------------------------------------------------------------------

def _research_brief_model():
    """Build the ResearchBrief Pydantic model lazily.

    We define it inside a function so the agent.py import is fast and
    pydantic is only required when this output mode is actually used.
    """
    from pydantic import BaseModel, Field

    class TableRow(BaseModel):
        model: str = Field(description="Model name (e.g. 'Transformer', 'BERT', 'GPT-3').")
        year: str = Field(description="Year (or year + month) the model was introduced.")
        org: str = Field(description="Organization that introduced it.")
        parameters: str = Field(description="Parameter count, or 'N/A' if not applicable.")

    class ResearchBrief(BaseModel):
        title: str = Field(description="Brief title heading.")
        introduction: str = Field(
            description=(
                "1-2 sentence opening paragraph that frames the topic. "
                "Synthesis based on observed facts."
            )
        )
        table_rows: list[TableRow] = Field(
            min_length=2,
            description="One row per source visited, in visit order.",
        )
        common_themes: str = Field(
            min_length=50,
            description=(
                "REQUIRED synthesis paragraph (>=50 chars) explaining patterns "
                "shared across the table_rows. This is NOT 'filling gaps with "
                "training knowledge' -- it's REQUIRED conclusion drawing from "
                "the facts you observed in this session."
            ),
        )

    return ResearchBrief


_OUTPUT_MODELS = {
    "research_brief": _research_brief_model,
}


def _save_artifacts(history, output_dir: Path, task: str, model: str,
                    base_url: str, wall_s: float, final_result: str) -> None:
    """Save agent_history.json + step_N.png + report.md to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full history JSON via browser-use's own serializer.
    history_path = output_dir / "agent_history.json"
    try:
        history.save_to_file(str(history_path))
    except Exception as exc:
        print(f"warn: could not save history: {exc}", file=sys.stderr)

    # Per-step screenshots. browser-use returns base64 strings.
    saved_screenshots: list[Path] = []
    try:
        shots = history.screenshots(return_none_if_not_screenshot=True)
    except Exception as exc:
        print(f"warn: could not enumerate screenshots: {exc}", file=sys.stderr)
        shots = []
    for i, shot_b64 in enumerate(shots, start=1):
        if not shot_b64:
            continue
        out = output_dir / f"step_{i:02d}.png"
        try:
            out.write_bytes(base64.b64decode(shot_b64))
            saved_screenshots.append(out)
        except Exception as exc:
            print(f"warn: could not save step_{i:02d}.png: {exc}",
                  file=sys.stderr)

    # Agent actions summary (one entry per step).
    actions: list[dict] = []
    try:
        actions = history.model_actions()
    except Exception:
        pass

    # Markdown report.
    report = output_dir / "report.md"
    lines = []
    lines.append(f"# browser-use run report")
    lines.append("")
    lines.append(f"- **Task**: {task}")
    lines.append(f"- **Model**: `{model}` via `{base_url}`")
    lines.append(f"- **Wall**: {wall_s:.1f}s")
    lines.append(f"- **Steps**: {len(actions)}")
    lines.append(f"- **Screenshots**: {len(saved_screenshots)} saved")
    lines.append("")
    lines.append("## Final result")
    lines.append("")
    lines.append("```")
    lines.append(final_result or "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("## Actions per step")
    lines.append("")
    for i, action in enumerate(actions, start=1):
        # action is a dict {action_name: action_params}; collapse for brevity.
        compact = json.dumps(action, default=str)[:200]
        lines.append(f"{i}. `{compact}`")
    lines.append("")
    lines.append("## Screenshots")
    lines.append("")
    for shot in saved_screenshots:
        lines.append(f"### {shot.name}")
        lines.append("")
        lines.append(f"![{shot.name}]({shot.name})")
        lines.append("")
    report.write_text("\n".join(lines))

    print(f"artifacts: {output_dir}/  "
          f"({len(saved_screenshots)} screenshots + agent_history.json + report.md)",
          file=sys.stderr)


async def run(task: str, *, model: str, base_url: str, api_key: str,
              max_steps: int, headless: bool,
              output_dir: Path, output_model_name: str = "") -> int:
    # Import here so --help is fast and import errors surface clearly.
    from browser_use import Agent, Browser, ChatOpenAI
    from browser_use.browser.profile import BrowserProfile

    output_model = None
    if output_model_name:
        builder = _OUTPUT_MODELS.get(output_model_name)
        if builder is None:
            available = ", ".join(sorted(_OUTPUT_MODELS)) or "(none)"
            print(f"ERROR: unknown output_model '{output_model_name}'. "
                  f"Available: {available}", file=sys.stderr)
            return 2
        output_model = builder()

    print(f"endpoint:   {base_url}", file=sys.stderr)
    print(f"model:      {model}", file=sys.stderr)
    print(f"headless:   {headless}", file=sys.stderr)
    print(f"output_dir: {output_dir}", file=sys.stderr)
    if output_model is not None:
        print(f"output_model: {output_model_name} "
              f"(Pydantic-enforced structured output)", file=sys.stderr)
    print(f"task:       {task[:200]}{'...' if len(task) > 200 else ''}",
          file=sys.stderr)

    # vllm-side notes (see sibling vllm preset README): a small VLM may
    # not produce perfectly-structured tool-use output. These flags loosen
    # the JSON schema enforcement so non-OpenAI providers don't trip.
    # temperature defaults to 0.0 to avoid the degenerate-decoding loops
    # we've observed on Qwen3.5-9B for long-form structured output tasks
    # (e.g. writing a research brief; the model gets trapped in repetition).
    # Override per-run via BROWSER_USE_TEMPERATURE if a task needs sampling
    # creativity.
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(_env("BROWSER_USE_TEMPERATURE", "0.0")),
        add_schema_to_system_prompt=True,
        remove_min_items_from_schema=True,
        remove_defaults_from_schema=True,
    )

    profile = BrowserProfile(
        headless=headless,
        # Inside Docker, chromium_sandbox defaults to False because the
        # Dockerfile sets IN_DOCKER=1 -- BrowserProfile's default reads
        # CONFIG.IN_DOCKER.
    )
    browser = Browser(browser_profile=profile)

    agent_kwargs = dict(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,  # screenshots per step -- requires a VLM
        max_actions_per_step=1,
        max_failures=3,
    )
    if output_model is not None:
        # NB: browser-use's Agent constructor calls this kwarg
        # `output_model_schema`, not `output_model`. The base class accepts
        # **kwargs, so wrong names get silently absorbed and ignored.
        agent_kwargs["output_model_schema"] = output_model

    agent = Agent(**agent_kwargs)

    started = time.monotonic()
    history = await agent.run(max_steps=max_steps)
    wall_s = time.monotonic() - started

    # Extract final result.
    final = ""
    try:
        final = history.final_result() or ""
    except Exception:
        pass
    if not final:
        try:
            final = str(history.history[-1])
        except Exception:
            final = "(no final result extracted)"

    border = "=" * 60
    print()
    print(border)
    print("browser-use final result:")
    print(border)
    print(final)
    print(border)

    # Save artifacts (best-effort; never fails the run).
    try:
        _save_artifacts(history, output_dir, task=task, model=model,
                        base_url=base_url, wall_s=wall_s,
                        final_result=final)
    except Exception as exc:
        print(f"warn: artifact save failed: {exc}", file=sys.stderr)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless browser-use agent against a local vLLM endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help="Natural-language task for the agent. Defaults to a "
                        "multi-step Wikipedia lookup (Linux release year).")
    p.add_argument("--model", default=_env("BROWSER_USE_MODEL", DEFAULT_MODEL),
                   help="Served model id (must match vllm's --served-model-name).")
    p.add_argument("--base-url", default=_env("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
                   help="OpenAI-compatible chat completions endpoint.")
    p.add_argument("--api-key", default=_env("VLLM_API_KEY", DEFAULT_API_KEY),
                   help="API key (vllm accepts any non-empty string).")
    p.add_argument("--max-steps", type=int,
                   default=int(_env("BROWSER_USE_MAX_STEPS", "20")),
                   help="Cap on agent reasoning steps.")
    p.add_argument("--output-dir",
                   default=_env("BROWSER_USE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
                   help="Where to save agent_history.json + step_N.png + report.md.")
    p.add_argument("--output-model",
                   default=_env("BROWSER_USE_OUTPUT_MODEL", ""),
                   choices=[""] + sorted(_OUTPUT_MODELS),
                   help="If set, wraps the agent's done() in a "
                        "StructuredOutputAction with this Pydantic schema. "
                        "Use 'research_brief' for the multi-source synthesis "
                        "task. Empty (default) uses browser-use's free-text "
                        "DoneAction.")
    p.add_argument("--no-headless", action="store_true",
                   help="Run with a visible browser window (requires X/Wayland; "
                        "doesn't work in the default Docker setup).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(
            task=args.task,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_steps=args.max_steps,
            headless=not args.no_headless,
            output_dir=Path(args.output_dir),
            output_model_name=args.output_model,
        ))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
