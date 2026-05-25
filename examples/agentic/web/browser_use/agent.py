#!/usr/bin/env python3
"""Headless browser-use agent pointed at a local OpenAI-compatible vLLM endpoint.

Reads the task from CLI args, talks to an LLM at $OPENCODE_BASE_URL with
api_key $VLLM_API_KEY, model $BROWSER_USE_MODEL (the served-model-name
from `cvl run vllm serve`). Browser is headless Chromium; the IN_DOCKER
env (set in the Dockerfile) makes browser-use auto-disable sandboxing
inside the container.

Default task targets httpbin.org/forms/post -- public, stable, deterministic
form that echoes the submitted values in the response so the smoke check
can grep for them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "sk-local"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TASK = (
    "Go to https://example.com and report the exact text content of "
    "the page's H1 heading. Reply with only the H1 text on a single "
    "line. Do not include any commentary, quotes, or formatting."
)


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


async def run(task: str, *, model: str, base_url: str, api_key: str,
              max_steps: int, headless: bool) -> int:
    # Import here so --help is fast and import errors surface clearly.
    from browser_use import Agent, Browser, ChatOpenAI
    from browser_use.browser.profile import BrowserProfile

    print(f"endpoint:   {base_url}", file=sys.stderr)
    print(f"model:      {model}", file=sys.stderr)
    print(f"headless:   {headless}", file=sys.stderr)
    print(f"task:       {task[:200]}{'...' if len(task) > 200 else ''}",
          file=sys.stderr)

    # vllm-side notes (see sibling vllm preset README): a small VLM may
    # not produce perfectly-structured tool-use output. These flags loosen
    # the JSON schema enforcement so non-OpenAI providers don't trip.
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
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

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,  # screenshots per step -- requires a VLM
        max_actions_per_step=1,
        max_failures=3,
    )

    history = await agent.run(max_steps=max_steps)
    # history is an AgentHistoryList -- final result text is in the last
    # step's evaluation / model output, helper exposes final_result().
    final = ""
    try:
        final = history.final_result() or ""
    except Exception:
        pass
    if not final:
        # Fallback: dump the last step text representation.
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
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless browser-use agent against a local vLLM endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help="Natural-language task for the agent. Defaults to a "
                        "httpbin.org/forms/post fill-and-submit smoke.")
    p.add_argument("--model", default=_env("BROWSER_USE_MODEL", DEFAULT_MODEL),
                   help="Served model id (must match vllm's --served-model-name).")
    p.add_argument("--base-url", default=_env("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
                   help="OpenAI-compatible chat completions endpoint.")
    p.add_argument("--api-key", default=_env("VLLM_API_KEY", DEFAULT_API_KEY),
                   help="API key (vllm accepts any non-empty string).")
    p.add_argument("--max-steps", type=int,
                   default=int(_env("BROWSER_USE_MAX_STEPS", "20")),
                   help="Cap on agent reasoning steps.")
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
