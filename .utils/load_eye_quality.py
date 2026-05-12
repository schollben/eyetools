import pandas as pd
from pathlib import Path


def load_eye_quality(analyzable_output_dir: Path) -> dict:
    df = pd.read_csv(analyzable_output_dir / "eye_data_quality.csv")

    mask = (df.trajectory == "left_eye_data_quality") & (df.component == "high_threshold")
    return {
        "EQtimestamps": df.loc[mask, "timestamp_s"].values,
        "EQframes":     df.loc[mask, "frame"].values,
        "LEQ": df.loc[(df.trajectory == "left_eye_data_quality") & (df.component == "high_threshold"), "value"].values.astype(bool),
        "REQ": df.loc[(df.trajectory == "right_eye_data_quality") & (df.component == "high_threshold"), "value"].values.astype(bool),
    }
