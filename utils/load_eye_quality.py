import numpy as np
import pandas as pd
from pathlib import Path


def load_eye_quality(analyzable_output_dir: Path, threshold: str) -> dict:

    df = pd.read_csv(analyzable_output_dir / "eye_data_quality.csv")

    mask = (df.trajectory == "left_eye_data_quality") & (df.component == f"{threshold}_threshold")

    leq = df.loc[(df.trajectory == "left_eye_data_quality") & (df.component == f"{threshold}_threshold"), "value"].to_numpy(dtype=bool)
    req = df.loc[(df.trajectory == "right_eye_data_quality") & (df.component == f"{threshold}_threshold"), "value"].to_numpy(dtype=bool)

    n = len(leq)
    leq_false_pct = 100.0 * (n - np.sum(leq)) / n if n else float("nan")
    req_false_pct = 100.0 * (n - np.sum(req)) / n if n else float("nan")
    print(f"[load_eye_quality] LEQ False: {leq_false_pct:.1f}%  REQ False: {req_false_pct:.1f}%  (n={n} frames)")
    
    #quick fix for now
    if req_false_pct == 100.0:
        req = leq
        print(f"using LEQ for REQ (100% false)")
    if leq_false_pct == 100.0:
        leq = req
        print(f"using REQ for LEQ (100% false)")

    return {
        "EQtimestamps": df.loc[mask, "timestamp_s"].values,
        "EQframes":     df.loc[mask, "frame"].values,
        "LEQ": leq,
        "REQ": req,
    }
