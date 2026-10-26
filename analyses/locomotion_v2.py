# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
#
# LOCOMOTION: does gaze-stabilization maturity gate running speed, or does running drive gaze maturation?
# Speed rises with EO, so comparisons are made at matched speed.
#
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
import warnings
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, AGE_COLORS,
                                       eo_groups, head_signal, eye_signal, running_mask, unwrap_deg,
                                       set_style, save_fig, plot_vs_eo)
from utils.find_arcs import find_arcs
from scipy.stats import chi2, pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MaxNLocator
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions
fig_tag = " delayed" if compare_delayed else ""   # appended to saved figure names

# OTHER SETTINGS
EXCLUDE = ((407, 11), (757, 15))   # (id, EO): ~30% head pose-flip artifact frames
flip_eye = "LE"          # LE horizontal is flipped to match RE, so both eyes can be pooled
gaze_pad = (12, 20)      # frames excluded before gaze-event onset / after its end (FIX)
nan_margin = 36          # frames excluded around each eye's tracking gaps (NaN), per eye
min_frames = 120         # frames (1 s) per session per speed bin
yaw_jump = 10            # deg/frame, unwrapped yaw step counted as a head pose flip
art_omega = 600          # deg/s, head angular speed counted as artifact
art_pad = 5              # frames padded around artifact frames
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
SPEED_EDGES = np.array([0, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600])   # mm/s, head linear speed bins
SPEED_CEN = (SPEED_EDGES[:-1] + SPEED_EDGES[1:]) / 2
STAT_MAX = 300           # mm/s, matched-speed LMMs use bins below this (every group has coverage)

Results = [R for R in Results if (R.id, R.eo) not in EXCLUDE]
if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")

# per-session frame masks
# ART = head artifact frames (yaw step > yaw_jump or |ω| > art_omega, ± art_pad)
# TURN = inside a find_arcs arc; STRAIGHT = outside arcs and outside head saccades (df_head onset -> peak)
# FIX = fixation frames (per eye): outside gaze shifts and away from tracking gaps
# RUN = running frames (body speed > speed_threshold for >= min_bout frames)
# STAT = stationary frames (body speed tracked and not running)
ART, ARCS, TURN, STRAIGHT, FIX, RUN, STAT = {}, {}, {}, {}, {}, {}, {}
for R in Results:
    n = len(R.yaw)
    art = (np.abs(np.diff(unwrap_deg(R.yaw), prepend=np.nan)) > yaw_jump) | (head_signal(R, "speed") > art_omega)
    ART[id(R)] = np.convolve(art, np.ones(2 * art_pad + 1), "same") > 0
    ARCS[id(R)] = find_arcs(R)
    turn, steer = np.zeros(n, bool), np.zeros(n, bool)
    for a, e in zip(ARCS[id(R)].onset, ARCS[id(R)].end):
        turn[a:e] = steer[a:e] = True
    for o, p in zip(R.df_head.onset, R.df_head.peak):
        steer[int(o):int(p) + 1] = True
    TURN[id(R)], STRAIGHT[id(R)] = turn, ~steer
    gs = np.zeros(len(R.LE_vx), bool)
    for df in (R.df_LEgaze, R.df_REgaze):
        for o, p in zip(df.onset, df.peak):
            gs[max(0, o - gaze_pad[0]):p + gaze_pad[1] + 1] = True
    FIX[id(R)] = {eye: ~gs & (np.convolve(~np.isfinite(eye_signal(R, eye, "vx")),
                                          np.ones(2 * nan_margin + 1), "same") == 0)
                  for eye in ("LE", "RE")}
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]

session_rows = {id(R): dict(id=R.id, eo=R.eo, group=title, group_idx=gi)
                for gi, (group, title) in enumerate(zip(groups, titles)) for R in group}   # per-session table, filled by each section
measure_families = {}


def speed_bin(R, mask):
    """SPEED_EDGES bin index per frame; -1 outside mask / untracked / >= last edge."""
    s = np.asarray(R.speed, float)
    b = np.digitize(s, SPEED_EDGES) - 1
    b[~np.isfinite(s) | (s >= SPEED_EDGES[-1]) | ~mask] = -1
    return b


def plot_bin_curves(ax, B, m, cen, label):
    """Mean ± SEM over sessions of per-(session × bin) measure m, per group; bins with >= 3 sessions."""
    for t, c in zip(titles, GROUP_COLORS):
        g = B[B.group == t].groupby("bin")[m]
        mu, se, nn = g.mean(), g.sem(), g.count()
        k = nn[nn >= 3].index
        ax.plot(cen[k], mu[k], color=c, marker="o", ms=3, mfc="w", lw=1, label=t)
        ax.fill_between(cen[k], (mu - se)[k], (mu + se)[k], color=c, alpha=0.2)
    ax.set_ylabel(label)
