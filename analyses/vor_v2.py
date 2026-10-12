# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle

import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from utils import create_subplot_grid, non_saccade_mask
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       LOCO_COLORS, eo_groups, head_signal, eye_signal, running_mask, clean_runs, run_xcorr,
                                       set_style, save_fig, plot_mean_se, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# Head signals: roll_v/pitch_v/yaw_v are stored as rad/s; head_signal converts to deg/s.
# Frame: flip_eye = "LE" is the HEAD frame — eye + = head yaw +, so VOR (counter-rotation) gain is
# NEGATIVE (-1 = full compensation). With "RE" the eye axis points the other way and the sign flips.
flip_eye = "LE"
sacc_pad = (12, 20)      # frames excluded before eye-saccade onset / after its end (see saccade-free cell)
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
min_frames = 500         # minimum saccade-free frames (both eyes pooled) for a session's gain fit
min_bin = 20             # minimum frames for a session's gain within one speed / head-velocity bin
speed_edges = np.array([0, 50, 100, 200, 400, 800])   # mm/s, gain vs running speed
mag_edges = np.arange(0, 220, 20)                      # deg/s, gain vs |head velocity|
max_lag = 15             # frames (125 ms), eye-head velocity cross-correlation
min_run = 4 * max_lag    # frames, shortest saccade-free run worth correlating

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% saccade-free frames and locomotor state
# METHODS: every eye-head measure below uses SACCADE-FREE frames only: frames outside every detected
# eye saccade of either eye (saccades are conjugate, so one eye's saccade excludes the frame for both),
# padded by sacc_pad[0] frames before onset and sacc_pad[1] frames after the saccade end (`peak`, the
# position turning point). Padding measured on isolated saccades: eye speed is above baseline from
# ~12 frames (100 ms) before onset to ~20 frames (167 ms) after the end (6 Hz filter ringing / overshoot).
# Head saccades are NOT excluded: compensatory eye movement not detected as a saccade stays in.
# METHODS: running = body speed > speed_threshold for >= min_bout frames (running_mask); stationary =
# body speed tracked and not running.
FREE, RUN, STAT = {}, {}, {}
for R in Results:
    FREE[id(R)] = non_saccade_mask(R, pad_pre=sacc_pad[0], pad_post=sacc_pad[1])
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]

print(pd.DataFrame([dict(id=R.id, eo=R.eo, free_pct=100 * FREE[id(R)].mean(),
                         free_stat_pct=100 * (FREE[id(R)] & STAT[id(R)]).mean(),
                         free_run_pct=100 * (FREE[id(R)] & RUN[id(R)]).mean())
                    for R in Results]).round(1).to_string(index=False))
