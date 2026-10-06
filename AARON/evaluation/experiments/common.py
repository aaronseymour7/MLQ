"""Shared setup for the evaluation experiments (read-only use of ../../j1j2_filter)."""
import os, sys, json, pathlib
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parents[1] / "j1j2_filter"
RES = HERE.parent / "results"
sys.path.insert(0, str(CODE))
import warnings; warnings.filterwarnings("ignore")

def save(name, obj):
    RES.mkdir(exist_ok=True)
    def conv(o):
        import numpy as np
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))
