import pandas as pd
from pathlib import Path

def load_toy_data(analyzable_output_dir: Path) -> dict:
    
    df = pd.read_csv(analyzable_output_dir / "toy_trajectories_resampled.csv")

    return {
        "toy_x": df.loc[(df.trajectory == "toy_top") & (df.component == "x"), "value"].values,
        "toy_y": df.loc[(df.trajectory == "toy_top") & (df.component == "y"), "value"].values,
    }
