"""Example single-case run (moved out of case_report.py, where it executed on import)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

from case_report import depolarizing_target, fake_target, ideal_target, run_case
from qiskit_ibm_runtime.fake_provider import FakeLagosV2


if __name__ == "__main__":
    rep = run_case(N=4, J2=0.0, eps=1e-2, which="emp",
                   targets=[ideal_target(), fake_target(FakeLagosV2()),
                            depolarizing_target(1e-3)],
                   shots=4000, out_dir="reports")
