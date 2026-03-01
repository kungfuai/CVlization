"""
Minimal exec-based REPL environment for Recursive Language Models.

Adapted from alexzhang13/rlm-minimal (MIT License).

The REPL:
- Executes Python code blocks with a persistent namespace across iterations
- Injects helper functions: llm_query(), peek(), FINAL_VAR()
- Captures stdout/stderr per execution
- Blocks dangerous builtins (eval, exec, globals, input)
"""

import sys
import io
import os
import tempfile
import threading
from contextlib import contextmanager


_SAFE_BUILTINS = {
    # Core types and functions
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "format": format, "hash": hash, "id": id,
    "any": any, "all": all, "hasattr": hasattr, "getattr": getattr,
    "setattr": setattr, "dir": dir,
    "iter": iter, "next": next, "pow": pow, "divmod": divmod,
    "bytes": bytes, "bytearray": bytearray, "callable": callable,
    "object": object, "property": property, "super": super,
    "staticmethod": staticmethod, "classmethod": classmethod,
    "slice": slice,
    "__import__": __import__, "open": open,
    # Exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
    "StopIteration": StopIteration, "OSError": OSError, "IOError": IOError,
    # Blocked — set to None so NameError is raised on use
    "eval": None, "exec": None, "compile": None,
    "globals": None, "locals": None, "input": None,
}


class REPLEnv:
    """
    Persistent Python REPL that the root LLM writes code into.

    State (variables) accumulates across calls to execute(). The LLM has access to:
      - `context`     : the full text loaded by load_context()
      - `peek(s, e)`  : character slice of context without printing the whole thing
      - `llm_query(p)`: call a sub-LLM with a plain text prompt
      - `FINAL_VAR(n)`: signal that variable `n` is the final answer
    """

    def __init__(self, llm_client):
        self._client = llm_client
        self._lock = threading.Lock()
        self._temp_dir = tempfile.mkdtemp(prefix="rlm_")
        self._globals: dict = {"__builtins__": _SAFE_BUILTINS}
        self._locals: dict = {}
        self._inject_helpers()

    def _inject_helpers(self):
        def llm_query(prompt: str) -> str:
            """Call a sub-LLM with a plain text prompt. Handles ~200K chars."""
            return self._client.completion([{"role": "user", "content": str(prompt)}])

        def peek(start: int = 0, end: int = 2000) -> str:
            """Return a character slice of the context string."""
            ctx = self._locals.get("context", "")
            return ctx[start:end]

        def FINAL_VAR(var_name: str) -> str:
            """Return the named REPL variable as a string (signals final answer)."""
            name = var_name.strip().strip("\"'")
            val = self._locals.get(name)
            return str(val) if val is not None else f"(variable '{name}' not found)"

        self._globals["llm_query"] = llm_query
        self._globals["peek"] = peek
        self._globals["FINAL_VAR"] = FINAL_VAR

    def load_context(self, text: str):
        """Write context to a temp file and load it as the `context` variable."""
        path = os.path.join(self._temp_dir, "context.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.execute(f"with open(r'{path}', encoding='utf-8') as _f:\n    context = _f.read()")

    def get_var(self, name: str):
        """Retrieve a variable from the REPL namespace."""
        return self._locals.get(name)

    @contextmanager
    def _redirect_io(self):
        """Thread-safe stdout/stderr capture."""
        with self._lock:
            old_out, old_err = sys.stdout, sys.stderr
            buf_out, buf_err = io.StringIO(), io.StringIO()
            sys.stdout = buf_out
            sys.stderr = buf_err
            try:
                yield buf_out, buf_err
            finally:
                sys.stdout = old_out
                sys.stderr = old_err

    def execute(self, code: str) -> tuple[str, str]:
        """
        Execute Python code in the persistent REPL namespace.

        Returns (stdout, stderr). Imports go into globals so they're
        available across iterations; other statements go into the combined
        locals/globals namespace.
        """
        old_cwd = os.getcwd()
        os.chdir(self._temp_dir)

        with self._redirect_io() as (buf_out, buf_err):
            try:
                # Split imports into globals so modules are accessible everywhere
                import_lines, other_lines = [], []
                for line in code.split("\n"):
                    stripped = line.lstrip()
                    if stripped.startswith(("import ", "from ")) and not stripped.startswith("#"):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                if import_lines:
                    exec("\n".join(import_lines), self._globals, self._globals)

                if other_lines:
                    ns = {**self._globals, **self._locals}
                    exec("\n".join(other_lines), ns, ns)
                    # Persist any new/updated variables to locals
                    g_keys = set(self._globals.keys())
                    for k, v in ns.items():
                        if k not in g_keys and not k.startswith("__"):
                            self._locals[k] = v

            except Exception as e:
                print(f"{type(e).__name__}: {e}", file=sys.stderr)

        os.chdir(old_cwd)
        return buf_out.getvalue(), buf_err.getvalue()

    def __del__(self):
        try:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
