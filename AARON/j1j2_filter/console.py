"""Console formatting helpers and quiet-run wrapper."""


import contextlib
import io
import numpy as np


VERBOSE = False                       # True: show all internal prints
WIDTH = 78


def banner(title, ch="="):
    print(f"\n{ch * WIDTH}\n{title}\n{ch * WIDTH}")


def section(title):
    print(f"\n{title}\n{'-' * WIDTH}")


def kv(items, ncol=2, lw=22, vw=16):
    """Aligned (label, value-string) pairs, ncol per line."""
    for i in range(0, len(items), ncol):
        print("  " + "".join(f"{l:<{lw}}{v:<{vw}}"
                             for l, v in items[i:i + ncol]).rstrip())


def sci(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}e}"


def fix(x, d=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def run_quiet(fn, *a, **kw):
    """Run fn, swallowing its prints (only lines containing 'warning' survive)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    text = buf.getvalue()
    if VERBOSE:
        print(text, end="")
    else:
        for line in text.splitlines():
            if "warning" in line.lower():
                print("  " + line.strip())
    return out
