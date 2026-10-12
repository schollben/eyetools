# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       eo_groups, head_signal, eye_signal, running_mask, clean_runs, run_xcorr,
                                       set_style, save_fig, plot_mean_se)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
flip_eye = "LE"          # LE horizontal is flipped to match RE, so both eyes can be pooled
gaze_pad = (12, 20)      # frames excluded before gaze-event onset / after its end (see fixation cell)
nan_margin = 36          # frames excluded around each eye's tracking gaps (NaN), per eye
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
vel_edges = np.arange(-300, 301, 20)    # deg/s, head velocity bins
pitch_edges = np.arange(-90, 91, 10)    # deg, head pitch bins
min_bin = 20             # minimum frames for a bin's median
min_frames = 500         # minimum fixation frames for a session's cross-correlation
max_lag = 15             # frames (125 ms), eye-head cross-correlation
min_run = 4 * max_lag    # frames, shortest fixation run worth correlating

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")
