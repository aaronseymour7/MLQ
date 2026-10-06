"""Run the full scaling study (resumable). Edit StudyConfig below."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root

import pipeline
from study import StudyConfig, a2a_routing, line_routing, run_study

cfg = StudyConfig(out_dir="study_v1",
                  J2_list=(0.0, 0.2411, 0.4),
                  N_resource=(4, 6, 8, 10, 12),   # N > 12 needs a non-ED spectrum source (uncertified)
                  N_ideal=(4, 6, 8, 10, 12),
                  eps_list=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
                  eps_cases=((6, 0.0), (8, 0.0), (6, 0.4)),
                  routing=(a2a_routing(), line_routing()))

if __name__ == "__main__":
    # ns=vars(pipeline): study looks up build_ctx, make_design, ... there
    # instead of in the notebook's __main__.
    out = run_study(cfg, ns=vars(pipeline))
    # out = run_study(cfg, compute_data=False)   # re-plot / re-report from cache only
