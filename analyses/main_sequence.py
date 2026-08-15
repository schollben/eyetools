# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import create_subplot_grid
import numpy as np
# plotting setup
import plotly.graph_objects as go
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA
# delayed vision: 416,411,403

#SESSION = getSesh.by_ferret(402, 420)    # multiple — preserves order by ferret
SESSION = getSesh.by_ferret(753)          # or load sessions from an inidividual ID
#SESSION = getSesh.by_name("session_2025-07-09_ferret_757_EyeCameras_P41_E13_analyzable_output") # or load a specific session by name
#SESSION = getSesh.by_name("session_2026-03-16_ferret_403_P49_E7_analyzable_output") # or load a specific session by name
#SESSION = getSesh.by_eo(7)      # or load sessions by a single EO number
#SESSION = getSesh.by_eo(10,20)   # or load sessions by an EO range (inclusive)

Results = []
for session in SESSION:
    
    R = load_session_data(session)
    removeBadData(R)
    process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=2,
                    velocity_threshold_head=2, min_duration=12, min_inter_event=12)
    Results.append(R)

n_sesh = len(Results)
print(n_sesh, "sessions loaded")
# to look at data execute: launch_viewer(load_session_data(SESSION[n]))
# or launch_viewer(R) if there is only 1 session in the list


# %% plots
# amplitude vs peak velocity (log-log), eyes combined

pool_by_eo = True  # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

if pool_by_eo:

    fig, axes = create_subplot_grid(len(eo_bins))
    groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
    titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]

else:

    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    pkv = np.concatenate([np.concatenate([R.df_LE["peak_velocity_deg_s"].to_numpy(),
                                          R.df_RE["peak_velocity_deg_s"].to_numpy()]) for R in group]).astype(float)

    x = np.log10(abs(amp))
    y = np.log10(abs(pkv))

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)


    
    ax.set_title(title)
    ax.axis([0.25, 1.75, 1.5, 3])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")


# amplitude vs duration (ms), eyes combined

pool_by_eo = True  # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

if pool_by_eo:
    fig, axes = create_subplot_grid(len(eo_bins))
    groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
    titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]
else:
    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    dur = np.concatenate([np.concatenate([(R.df_LE["peak"] - R.df_LE["onset"]).to_numpy(),
                                          (R.df_RE["peak"] - R.df_RE["onset"]).to_numpy()]) for R in group]).astype(float)
    dur_ms = dur / 120.0 * 1000.0

    inds = np.isfinite(dur_ms)
    x = abs(amp[inds])
    y = dur_ms[inds]

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)
    
    ax.set_title(title)
    ax.axis([0, 40, 0, 500])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Duration (ms)")
