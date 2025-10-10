"""
Reward functions for GRPO training - matrix multiplication kernel generation.
Extracted from: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
"""

import numpy as np
import ast
import sys
import sysconfig
import types
import os
import gc
import time
import statistics
import signal
from pathlib import Path
from contextlib import contextmanager


# ============================================================================
# Helper Functions
# ============================================================================

def generate_random_matrices(seed=3407, n=256):
    random_state = np.random.RandomState(seed)
    n, k, m = random_state.randint(1, n+1, size=3)
    A = np.random.uniform(-10, 10, size=(n, k))
    B = np.random.uniform(-10, 10, size=(k, m))
    return A, A.tolist(), B, B.tolist()


def calculate_difference(pred, real):
    if pred is None:
        return 5, 5
    assert real is not None
    try:
        difference = pred - real
    except:
        return 5, 5
    amax_error = float(np.amax(difference))
    mse_error = float(np.mean(np.square(difference)))
    return amax_error, mse_error


def _stdlib_names():
    """
    Build a set of canonical stdlib top-level module/package names.
    """
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__")

    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        pass

    return names


_STDLIB_SET = _stdlib_names()


def check_only_stdlib_imports(code: str):
    """
    Return (ok: bool, details: dict)
    ok == True -> all absolute imports are from the stdlib.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom):
            nonlocal relative_count
            if (node.level or 0) > 0:
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }


def create_locked_down_function(function):
    output_function = {}
    exec(function, {}, output_function)
    new_matmul = output_function["matmul"]
    new_matmul = types.FunctionType(new_matmul.__code__, {})
    return new_matmul


def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def matmul(A, B):"):
            return fx
    return None


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


class Benchmarker:
    def __init__(self, trials=3, loops=1, timeout=30):
        self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype=np.uint8)
        self.trials = trials
        self.loops = loops
        assert timeout > 0
        self.timeout = timeout

    def thrash(self):
        self.buffer ^= 1
        return int(self.buffer[::4096].sum())

    def benchmark(self, function, arguments):
        assert len(arguments) == self.loops
        samples = []
        exceptions = []
        timed_out = 0
        for _ in range(self.trials):
            gc.collect()
            gc.disable()
            self.thrash()
            t_start = time.perf_counter_ns()
            for i in range(self.loops):
                try:
                    with time_limit(self.timeout):
                        function(*arguments[i])
                except TimeoutError as e:
                    timed_out += 1
                except Exception as e:
                    exceptions.append(str(e))
            t_end = time.perf_counter_ns()
            gc.enable()
            samples.append((t_end - t_start) // max(1, self.loops))
        return {
            "median_ns": int(statistics.median(samples)),
            "mean_ns": int(statistics.fmean(samples)),
            "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
            "exceptions": exceptions,
            "timeouts": timed_out,
        }


# Global benchmarker instance
benchmarker = Benchmarker(trials=3, loops=1, timeout=30)


# ============================================================================
# Reward Functions
# ============================================================================

def function_works(completions, **kwargs):
    """Check if the generated function can be executed without errors."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        print(function)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_matmul = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores


def no_cheating(completions, **kwargs):
    """Heavily penalize using non-stdlib imports (like numpy in the solution)."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        else:
            ok = False
        scores.append(1.0 if ok else -20.0)
    return scores


def correctness_check(completions, **kwargs):
    """Check if the function produces correct results."""
    scores = []
    A, A_list, B, B_list = generate_random_matrices(seed=np.random.randint(10000), n=128)
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        try:
            pred = new_matmul(A_list.copy(), B_list.copy())
        except:
            scores.append(-2.0)
            continue
        true = np.matmul(A, B)
        amax_error, mse_error = calculate_difference(pred, true)

        machine_epsilon = 100 * np.finfo(np.float64).eps
        if amax_error >= 3:
            score = -3.0
        elif amax_error >= 2:
            score = -2.5
        elif amax_error >= 1:
            score = -2.0
        elif amax_error >= 0.5:
            score = -1.0
        elif amax_error >= 100 * machine_epsilon:
            score = 0.0
        elif amax_error >= machine_epsilon:
            score = 1.0
        else:
            score = 3.0

        if mse_error >= 3:
            score += -3.0
        elif mse_error >= 2:
            score += -2.5
        elif mse_error >= 1:
            score += -2.0
        elif mse_error >= 0.5:
            score += -1.0
        elif mse_error >= 100 * machine_epsilon:
            score += 0.0
        elif mse_error >= machine_epsilon:
            score += 1.0
        else:
            score += 3.0
        scores.append(score)
    return scores


def speed_check(completions, **kwargs):
    """Check if the function is faster than numpy's matmul."""
    scores = []
    A, A_list, B, B_list = generate_random_matrices(seed=np.random.randint(10000), n=256)
    numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        new_results = benchmarker.benchmark(new_matmul, [(A_list.copy(), B_list.copy())])

        negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
        positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
        score = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
        if score >= 10:
            score = 10
        if score <= -10:
            score = -10
        scores.append(score)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return scores
