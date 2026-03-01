#!/usr/bin/env python3
"""
Needle-in-haystack demo using Recursive Language Models (RLM).

The LLM writes Python code in a REPL loop to search a large context
for a hidden magic number. The context never fits in the prompt window —
the LLM uses the REPL to programmatically explore it.

Based on the RLM paradigm from:
  - alexzhang13/rlm-minimal (MIT)
  - "Recursive Language Models" (Zhang, Kraska, Khattab 2024)

Environment variables:
  LLM_BACKEND     anthropic (default) or openai
  MODEL           claude-haiku-4-5-20251001 (default) or any model string
  ANTHROPIC_API_KEY / OPENAI_API_KEY
  CONTEXT_LINES   number of lines to generate (default: 100000)
  MAX_ITERATIONS  max RLM loop iterations (default: 10)
"""

import os
import re
import sys
import random
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

from llm_client import LLMClient
from rlm_repl import REPLEnv


# ---------------------------------------------------------------------------
# System prompt: teaches the LLM how to use the REPL environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are solving a needle-in-haystack task using a Python REPL.
A large text context has been loaded into the variable `context` in your REPL.
Somewhere in that text is a line containing a magic number.

**Available REPL tools:**
- `context`              — the full text string (may be millions of characters)
- `peek(start, end)`     — inspect a character slice (default: first 2000 chars)
- `llm_query(prompt)`    — call a sub-LLM with a text prompt (handles ~200K chars)
- Full Python standard library (re, io, etc.)

**How to use the REPL:**
Write Python code inside ```repl``` blocks. State persists between iterations.

Example — fast regex search:
```repl
import re
match = re.search(r'magic number is (\\d+)', context)
if match:
    answer = match.group(1)
    print(f"Found: {answer}")
```

Example — chunk-based search with sub-LLM:
```repl
chunk_size = 200_000
for i in range(0, len(context), chunk_size):
    chunk = context[i:i+chunk_size]
    if 'magic' in chunk.lower():
        answer = llm_query(f"Find the magic number in this text:\\n{chunk}")
        print(f"Sub-LLM says: {answer}")
        break
```

**To finish:**
Once you have found and verified the magic number, use ONE of:
- `FINAL("the number")` — provide the answer directly
- `FINAL_VAR("variable_name")` — return a REPL variable as the final answer

Do NOT use FINAL until you have verified the answer in the REPL.
Think step-by-step and execute code immediately rather than describing what you'll do.
"""


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

def generate_context(num_lines: int, magic_number: str) -> tuple[str, int]:
    """Generate synthetic text with a magic number hidden at a random line."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "data", "text", "random", "sample", "content", "line", "filler",
        "some", "more", "words", "here", "test", "value", "entry",
    ]
    lines = [
        " ".join(random.choices(words, k=random.randint(4, 8)))
        for _ in range(num_lines)
    ]
    needle_pos = random.randint(num_lines // 4, 3 * num_lines // 4)
    lines[needle_pos] = f"The magic number is {magic_number}"
    return "\n".join(lines), needle_pos


# ---------------------------------------------------------------------------
# RLM response parsing
# ---------------------------------------------------------------------------

def find_code_blocks(text: str) -> list[str]:
    return re.findall(r"```repl\s*\n(.*?)\n```", text, re.DOTALL)


def find_final_answer(text: str) -> tuple[str, str] | None:
    """Return (kind, content) where kind is 'VAR' or 'INLINE', or None."""
    # FINAL_VAR takes priority
    m = re.search(r"^\s*FINAL_VAR\(([\"']?)(\w+)\1\)", text, re.MULTILINE)
    if m:
        return ("VAR", m.group(2))
    m = re.search(r"^\s*FINAL\(([\"']?)(.*?)([\"']?)\)", text, re.MULTILINE | re.DOTALL)
    if m:
        return ("INLINE", m.group(2).strip())
    return None


# ---------------------------------------------------------------------------
# Main RLM loop
# ---------------------------------------------------------------------------

def run_rlm(
    context: str,
    query: str,
    client: LLMClient,
    repl: REPLEnv,
    max_iterations: int,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(max_iterations):
        if i == 0:
            user_msg = (
                f"The context has been loaded into the REPL (variable: `context`).\n"
                f"Context size: {len(context):,} characters, "
                f"{context.count(chr(10)) + 1:,} lines.\n\n"
                f'Your task: "{query}"\n\n'
                f"Start by searching the context in the REPL."
            )
        else:
            user_msg = (
                f'Continue solving: "{query}"\n'
                f"Use the REPL to find and verify the magic number, "
                f"then call FINAL() or FINAL_VAR()."
            )

        messages.append({"role": "user", "content": user_msg})

        print(f"\n[Iter {i + 1}/{max_iterations}] Calling {client.backend}/{client.model}...")
        response = client.completion(messages)
        messages.append({"role": "assistant", "content": response})

        preview = response[:280].replace("\n", " ")
        print(f"  LLM: {preview}{'...' if len(response) > 280 else ''}")

        # Execute all code blocks in order
        blocks = find_code_blocks(response)
        for code in blocks:
            code_preview = code[:80].replace("\n", "; ")
            print(f"  [REPL] {code_preview}{'...' if len(code) > 80 else ''}")

            stdout, stderr = repl.execute(code)
            combined = ""
            if stdout.strip():
                print(f"  stdout: {stdout.strip()[:300]}")
                combined += stdout
            if stderr.strip():
                print(f"  stderr: {stderr.strip()[:200]}")
                combined += f"\nError: {stderr}"
            if not combined.strip():
                combined = "(no output)"

            messages.append({
                "role": "user",
                "content": f"REPL output:\n{combined[:4000]}",
            })

        # Check for final answer signal
        result = find_final_answer(response)
        if result:
            kind, content = result
            if kind == "VAR":
                val = repl.get_var(content)
                if val is not None:
                    return str(val)
                print(f"  Warning: FINAL_VAR('{content}') — variable not found, continuing...")
            else:
                return content

    # Max iterations reached — ask explicitly
    print(f"\n[Max iterations reached] Requesting final answer...")
    messages.append({
        "role": "user",
        "content": "Provide your final answer now using FINAL(the_number).",
    })
    response = client.completion(messages)
    result = find_final_answer(response)
    if result:
        kind, content = result
        if kind == "VAR":
            return str(repl.get_var(content) or content)
        return content

    # Last resort: extract any 7-digit number from the response
    nums = re.findall(r"\b\d{7}\b", response)
    return nums[0] if nums else response.strip()[:100]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    num_lines = int(os.environ.get("CONTEXT_LINES", "100000"))
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "10"))
    magic_number = str(random.randint(1_000_000, 9_999_999))

    print("=== RLM Needle-in-Haystack Demo ===")
    print(f"Context:    {num_lines:,} lines")
    print(f"Backend:    {os.environ.get('LLM_BACKEND', 'anthropic')}")
    print(f"Model:      {os.environ.get('MODEL', 'claude-haiku-4-5-20251001')}")
    print(f"Max iters:  {max_iterations}")

    print(f"\nGenerating context...", end=" ", flush=True)
    context, needle_pos = generate_context(num_lines, magic_number)
    print(f"done ({len(context):,} chars, needle at line {needle_pos:,})")
    print(f"Magic number: {magic_number} [hidden from LLM]")

    client = LLMClient()
    repl = REPLEnv(client)
    repl.load_context(context)

    query = "I'm looking for a magic number hidden somewhere in the context. What is it?"

    print("\nStarting RLM loop...")
    answer = run_rlm(context, query, client, repl, max_iterations)

    found = magic_number in str(answer)
    print(f"\n{'=' * 40}")
    print(f"Answer:   {answer}")
    print(f"Expected: {magic_number}")
    print(f"Result:   {'PASS' if found else 'FAIL'}")
    print(f"{'=' * 40}")

    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
