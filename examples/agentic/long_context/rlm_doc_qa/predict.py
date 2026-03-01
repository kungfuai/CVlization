#!/usr/bin/env python3
"""
Long document QA using Recursive Language Models (RLM).

The LLM answers questions about documents too large to fit in a single prompt.
It writes Python in a REPL loop to chunk, delegate to sub-LLMs, and synthesize.

Supports plain text, markdown, source code, and any UTF-8 file.

Environment variables:
  DOCUMENT_PATH   path to the document to load (required, or use --document)
  QUERY           question to answer (required, or use --query)
  LLM_BACKEND     anthropic (default) or openai
  MODEL           claude-haiku-4-5-20251001 (default) or any model string
  SUB_MODEL       model for sub-LLM calls (defaults to MODEL)
  ANTHROPIC_API_KEY / OPENAI_API_KEY
  MAX_ITERATIONS  max RLM loop iterations (default: 15)
  CHUNK_SIZE      max chars per sub-LLM chunk (default: 150000)
"""

import os
import re
import sys
import logging
import warnings
import argparse
import textwrap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

from llm_client import LLMClient
from rlm_repl import REPLEnv


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are answering a question about a document that is too large to fit in a single prompt.
The document has been loaded into the REPL variable `context`. You must use the REPL to read and analyze it.

IMPORTANT: To run Python code, you MUST wrap it in triple-backtick repl blocks like this:
```repl
print("hello")
```
Do NOT use XML, JSON, or any other format. Only the ```repl...``` markdown format is supported.

**Available REPL tools:**
- `context`                   — the full document text
- `doc_len`                   — total character count of the document
- `peek(start, end)`          — inspect a character slice (e.g. peek(0, 3000))
- `llm_query(prompt)`         — call a sub-LLM with a text prompt (~150K char capacity)
- `chunk_context(size)`       — returns list of (start, end) tuples splitting context into chunks
- Full Python standard library (re, json, etc.)

**Recommended strategy:**

1. Peek at the start to understand document structure:
```repl
print(peek(0, 3000))
print(f"Total length: {doc_len:,} chars")
```

2. Choose a chunking approach (by size, by section, by regex split):
```repl
import re
# Option A: fixed-size chunks
chunks_idx = chunk_context(150000)
print(f"{len(chunks_idx)} chunks")

# Option B: split by section headers
sections = re.split(r'\\n#{1,3} ', context)
print(f"{len(sections)} sections")
```

3. Query sub-LLMs per chunk, accumulate findings:
```repl
findings = []
for start, end in chunk_context(150000):
    chunk = context[start:end]
    prompt = f"Document excerpt ({start}-{end} of {doc_len} chars):\\n{chunk}\\n\\nQuestion: {query}\\nAnswer only from this excerpt. If not found, say so."
    result = llm_query(prompt)
    findings.append(result)
    print(f"Chunk {start//150000+1}: {result[:80]}")
```

4. Synthesize and return the final answer:
```repl
joined = "\\n---\\n".join(findings)
synthesis_prompt = f"Based on these findings from different parts of the document, answer: {query}\\n\\nFindings:\\n{joined}"
final_answer = llm_query(synthesis_prompt)
print(final_answer)
```

Then call: FINAL_VAR("final_answer")

**Rules:**
- The variable `query` holds the user's question — reference it in sub-LLM prompts
- Do NOT use FINAL until you have synthesized across the whole document
- Use `llm_query` for any analysis that requires understanding language
- Use Python (re, split, etc.) for structural operations (counting, slicing, finding headers)
- Think step-by-step, execute immediately rather than just describing what you'll do
"""


# ---------------------------------------------------------------------------
# RLM parsing helpers
# ---------------------------------------------------------------------------

def find_code_blocks(text: str) -> list[str]:
    # Standard markdown repl blocks (preferred)
    blocks = re.findall(r"```repl\s*\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        return blocks
    # Fallback: XML tool-call format some models emit
    xml_blocks = re.findall(
        r'<invoke[^>]*name=["\']?repl["\']?[^>]*>.*?<parameter[^>]*name=["\']?code["\']?[^>]*>(.*?)</parameter>',
        text, re.DOTALL
    )
    return xml_blocks


def find_final_answer(text: str) -> tuple[str, str] | None:
    m = re.search(r"^\s*FINAL_VAR\(([\"']?)(\w+)\1\)", text, re.MULTILINE)
    if m:
        return ("VAR", m.group(2))
    m = re.search(r"^\s*FINAL\(([\"']?)(.*?)([\"']?)\)", text, re.MULTILINE | re.DOTALL)
    if m:
        return ("INLINE", m.group(2).strip())
    return None


# ---------------------------------------------------------------------------
# RLM loop
# ---------------------------------------------------------------------------

def run_rlm(
    doc_text: str,
    query: str,
    client: LLMClient,
    repl: REPLEnv,
    max_iterations: int,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(max_iterations):
        if i == 0:
            user_msg = (
                f"The document has been loaded into `context` in the REPL.\n"
                f"Document size: {len(doc_text):,} characters.\n\n"
                f'Question to answer: "{query}"\n\n'
                f"Start by peeking at the document structure, then chunk and analyze it."
            )
        else:
            user_msg = (
                f'Continue answering: "{query}"\n'
                f"Keep using the REPL to analyze the document. "
                f"When you have a complete answer, call FINAL_VAR(\"final_answer\") or FINAL(\"...\")."
            )

        messages.append({"role": "user", "content": user_msg})

        print(f"\n[Iter {i + 1}/{max_iterations}] Calling {client.backend}/{client.model}...")
        response = client.completion(messages)
        messages.append({"role": "assistant", "content": response})

        preview = response[:300].replace("\n", " ")
        print(f"  LLM: {preview}{'...' if len(response) > 300 else ''}")

        blocks = find_code_blocks(response)
        for code in blocks:
            code_preview = code[:80].replace("\n", "; ")
            print(f"  [REPL] {code_preview}{'...' if len(code) > 80 else ''}")

            stdout, stderr = repl.execute(code)
            combined = ""
            if stdout.strip():
                out_preview = stdout.strip()[:500]
                print(f"  stdout: {out_preview}")
                combined += stdout
            if stderr.strip():
                print(f"  stderr: {stderr.strip()[:200]}")
                combined += f"\nError: {stderr}"
            if not combined.strip():
                combined = "(no output)"

            messages.append({
                "role": "user",
                "content": f"REPL output:\n{combined[:5000]}",
            })

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

    # Max iterations reached
    print(f"\n[Max iterations reached] Requesting final answer...")
    messages.append({
        "role": "user",
        "content": "Provide your final answer now using FINAL(\"your answer here\").",
    })
    response = client.completion(messages)
    result = find_final_answer(response)
    if result:
        kind, content = result
        if kind == "VAR":
            return str(repl.get_var(content) or content)
        return content
    return response.strip()[:500]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Long document QA using RLM")
    parser.add_argument("--document", default=os.environ.get("DOCUMENT_PATH"),
                        help="Path to the document (or set DOCUMENT_PATH)")
    parser.add_argument("--query", default=os.environ.get("QUERY"),
                        help="Question to answer (or set QUERY)")
    args = parser.parse_args()

    if not args.document:
        print("Error: provide --document or set DOCUMENT_PATH", file=sys.stderr)
        sys.exit(1)
    if not args.query:
        print("Error: provide --query or set QUERY", file=sys.stderr)
        sys.exit(1)

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "15"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "150000"))

    print("=== RLM Document QA ===")
    print(f"Document:   {args.document}")
    print(f"Query:      {args.query}")
    print(f"Backend:    {os.environ.get('LLM_BACKEND', 'anthropic')}")
    print(f"Model:      {os.environ.get('MODEL', 'claude-haiku-4-5-20251001')}")
    print(f"Max iters:  {max_iterations}")

    print(f"\nLoading document...", end=" ", flush=True)
    doc_text = load_document(args.document)
    print(f"done ({len(doc_text):,} chars)")

    client = LLMClient()
    repl = REPLEnv(client)

    # Inject document-specific helpers into the REPL
    repl.load_context(doc_text)
    repl.execute(f"doc_len = len(context)")
    repl.execute(f"query = {repr(args.query)}")
    repl.execute(textwrap.dedent(f"""
        def chunk_context(size={chunk_size}):
            '''Return list of (start, end) character index tuples.'''
            total = len(context)
            return [(i, min(i + size, total)) for i in range(0, total, size)]
    """).strip())

    print("\nStarting RLM loop...")
    answer = run_rlm(doc_text, args.query, client, repl, max_iterations)

    print(f"\n{'=' * 60}")
    print(f"Question: {args.query}")
    print(f"{'=' * 60}")
    print(answer)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
