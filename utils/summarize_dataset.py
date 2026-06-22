import os
import sys
from pathlib import Path

# ── Configure for your machine ────────────────────────────────────────────────
os.environ.setdefault("EYETOOLS_DATA_DIR",
    "/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/VisBehavDev/data/analyzable_outputs")
os.environ.setdefault("EYETOOLS_PYTHON_CODE_DIR",
    "/Users/benjaminscholl/Documents/bs")
PLOT = True   # set to False to skip the figure
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

from config import DATA_DIR
from parse_session_name import parse_session_name


def _read_eye_quality(session_dir: Path):
    csv = session_dir / "eye_data_quality.csv"
    if not csv.exists():
        return float("nan"), float("nan"), float("nan")
    try:
        df = pd.read_csv(csv)
        leq = df.loc[
            (df.trajectory == "left_eye_data_quality") & (df.component == "low_threshold"),
            "value",
        ].to_numpy(dtype=bool)
        req = df.loc[
            (df.trajectory == "right_eye_data_quality") & (df.component == "low_threshold"),
            "value",
        ].to_numpy(dtype=bool)
        n = len(leq)
        if n == 0:
            return float("nan"), float("nan"), float("nan")
        return n, float(np.mean(leq)), float(np.mean(req))
    except Exception:
        return float("nan"), float("nan"), float("nan")


def summarize_dataset(plot: bool = False):
    if not DATA_DIR.exists():
        print(
            f"\nWarning: data directory not found:\n  {DATA_DIR}\n"
            "Please update EYETOOLS_DATA_DIR (or the hardcoded path at the top of this file) "
            "to match your machine.\n"
        )
        return None

    rows = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            parsed = parse_session_name(d.name)
        except Exception:
            continue
        n_frames, frac_good_LE, frac_good_RE = _read_eye_quality(d)
        rows.append(
            {
                "session":      parsed["session"],
                "date":         parsed["date"],
                "ferret_id":    parsed["id"],
                "age_days":     parsed["age"],
                "eo":           parsed["eo"],
                "n_frames":     n_frames,
                "frac_good_LE": frac_good_LE,
                "frac_good_RE": frac_good_RE,
            }
        )

    df = pd.DataFrame(rows).sort_values(["ferret_id", "eo"]).reset_index(drop=True)

    n_sessions   = len(df)
    n_animals    = df["ferret_id"].nunique()
    ferret_list  = [int(x) for x in sorted(df["ferret_id"].unique())]
    eo_min       = int(df["eo"].min())
    eo_max       = int(df["eo"].max())
    total_frames = int(df["n_frames"].dropna().sum())

    print(f"Sessions   : {n_sessions}")
    print(f"Animals    : {n_animals}  (IDs: {ferret_list})")
    print(f"EO range   : EO{eo_min} – EO{eo_max}")
    print(f"Total frames: {total_frames:,}")

    if not plot:
        return df

    ferret_ids = ferret_list
    colors = {fid: cm.tab10(i / max(len(ferret_ids) - 1, 1)) for i, fid in enumerate(ferret_ids)}
    # sub-filter df to use plain int keys consistently
    df["ferret_id"] = df["ferret_id"].astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
    fig.suptitle(
        f"Sessions: {n_sessions}   |   Animals: {n_animals} {ferret_list}\n"
        f"EO range: EO{eo_min}–EO{eo_max}   |   Total frames: {total_frames:,}",
        fontsize=7, ha="left", x=0.05,
    )
    fig.tight_layout(pad=3)
    sns.despine(fig=fig)

    # background EO-range shading (top panel only)
    axes[0].axvspan(-0.5, 7.5,  facecolor="#d0d0d0", alpha=0.35, zorder=0)
    axes[0].axvspan(7.5,  15.5, facecolor="#888888", alpha=0.25, zorder=0)
    # horizontal reference line at 10 000 frames
    axes[0].axhline(10000, color="#888888", linewidth=0.8, linestyle="--", zorder=1)

    for fid in ferret_ids:
        sub = df[df["ferret_id"] == fid].sort_values("eo")
        c = colors[fid]

        # panel 0: frames — connected line + filled circles
        axes[0].plot(sub["eo"], sub["n_frames"], color=c, linewidth=0.8, alpha=0.5, zorder=2)
        axes[0].scatter(sub["eo"], sub["n_frames"], color=c, s=40, alpha=0.6,
                        edgecolors="white", linewidths=0.4, zorder=3)

        # panel 1: bad frames — LE filled, RE open
        y_le = 1 - sub["frac_good_LE"]
        y_re = 1 - sub["frac_good_RE"]
        axes[1].scatter(sub["eo"], y_le, color=c, s=40, alpha=0.6,
                        edgecolors="white", linewidths=0.4, zorder=3)
        axes[1].scatter(sub["eo"], y_re, facecolors="white", edgecolors=c,
                        s=40, alpha=0.8, linewidths=0.8, zorder=3)

    axes[0].set_ylabel("Frames per session")
    axes[1].set_ylabel("Fraction bad frames")
    axes[1].set_xlabel("EO day")

    ferret_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[fid],
                   markersize=6, label=f"F{fid}")
        for fid in ferret_ids
    ]
    eye_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="white", markersize=6, label="LE"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="gray", markersize=6, label="RE"),
    ]
    axes[1].legend(handles=ferret_handles + eye_handles, fontsize=6, frameon=False,
                   title="Ferret / Eye", title_fontsize=6, loc="upper right")

    return df, fig


if __name__ == "__main__":
    result = summarize_dataset(plot=PLOT)
    if result is None:
        sys.exit(1)
    df = result[0] if isinstance(result, tuple) else result
    print(df.to_string())
    if PLOT:
        plt.show()
