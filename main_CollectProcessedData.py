# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init

%load_ext autoreload
%autoreload
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import saccade_triggered_average, saccade_triggered_average, saccade_andHead_triggered_average
from utils import eye_head_correlogram, eye_head_coincidence, plot_eye_head_correlogram
from utils import create_subplot_grid
import numpy as np
from data_viewer import launch_viewer
# plotting setup
import plotly.graph_objects as go
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# %% load data, delayed vision: 416,411,403
#SESSION = getSesh.by_ferret(402, 420)    # multiple — preserves order by ferret
#SESSION = getSesh.by_ferret(420)          # or load sessions from an inidividual ID
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


# %% BASIC PLOTS: saccade-triggered average of eye position and head rotation

fig = saccade_triggered_average(Results, window=30, binocular="separate")
# plt.savefig(f'{SAVELOC}/LE_pos_eo.svg', format='svg', bbox_inches='tight')

#note: head saccades events are ANCHORED to initiation of head movement
fig = saccade_andHead_triggered_average(Results, window=120, binocular="separate")
# plt.savefig(f'{SAVELOC}/head_and_LE_pos_eo.svg', format='svg', bbox_inches='tight')


# %% BASIC PLOTS: pupil distribution across sessions

fig, axes = create_subplot_grid(n_sesh)

thresh_for_bad_data = 6 #what is happening when pupil size is > 6 mm? 
# (maybe the eye is closed and the pupil is not visible, so the software is returning a large value for pupil size)

for n in range(n_sesh):

    pL = Results[n].LE_pupil[ Results[n].LE_pupil < thresh_for_bad_data ]
    pR = Results[n].RE_pupil[ Results[n].RE_pupil < thresh_for_bad_data ]

    sns.histplot(pL, ax=axes[n], color="#725EE7", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
    sns.histplot(pR, ax=axes[n], color="#E93115", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
    
    # axes[n].set_xlabel("Pupil size (mm)")
    axes[n].set_title(f"EO: {Results[n].eo}")
    axes[n].axis([1.5, 6.5, 0, 0.2])  # [xmin, xmax, ymin, ymax]
    axes[n].set_xticks([2, 4, 6])
    axes[n].set_yticks([0, 0.2])

    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


# %% BASIC PLOTS: saccade rate distribution
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    pL = Results[n].eyeRateLE
    pR = Results[n].eyeRateRE

    sns.histplot(pL, ax=axes[n], color="#725EE7", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    sns.histplot(pR, ax=axes[n], color="#E93115", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    axes[n].axis([0, 4, 0, 0.5])  # [xmin, xmax, ymin, ymax]

    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


# %% BASIC PLOTS: gaze rate distribution
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    pL = Results[n].gazeRateLE
    pR = Results[n].gazeRateRE

    sns.histplot(pL, ax=axes[n], color="#725EE7", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    sns.histplot(pR, ax=axes[n], color="#E93115", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    axes[n].axis([0, 4, 0, 0.5])  # [xmin, xmax, ymin, ymax]

    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


# %% BASIC PLOTS: speed distribution (mm/s)
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    dat = Results[n].speed
    sns.histplot(dat, ax=axes[n], color="#000000", stat="probability", binwidth=20, linewidth=0, alpha=0.6)
    # axes[n].set_xscale("log") #need to remove 0s first
    
    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


# %% BASIC PLOT: speed vs angular velocity of head rotations (total ang vel)
# log-log plot here
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    inds = (Results[n].speed > 0) & (np.abs( Results[n].angVelocities) > 0)
    x = Results[n].speed[inds]
    y = np.abs( Results[n].angVelocities[inds])
    x = np.log10(x)
    y = np.log10(y)
    sns.scatterplot(ax=axes[n], x=x, y=y, s=1, alpha=0.1)
    # axes[n].axis([1, 800, 0, 50])  # [xmin, xmax, ymin, ymax]

    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


# %% 2D scatters comparing speed and pupil size distributions between ages (young to old)

fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    inds = (Results[n].speed > 0) & (np.abs(Results[n].LE_pupil) < 6) & (np.abs(Results[n].RE_pupil) < 6)
    x = Results[n].speed[inds]
    y1 = np.abs(Results[n].LE_pupil[inds])
    y2 = np.abs(Results[n].RE_pupil[inds])
    sns.scatterplot(ax=axes[n], x=x, y=y1, s=1, alpha=0.1)
    sns.scatterplot(ax=axes[n], x=x, y=y2, s=1, alpha=0.1)

    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


 # %% examine binocular coordination of eye movements
from scipy.stats import spearmanr

fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    velocity_threshold = 20 #threshold when eye movement is happening?
    inds = (Results[n].LE_vx > velocity_threshold) & (Results[n].RE_vx > velocity_threshold)
    x = Results[n].LE_x[inds]
    y = -Results[n].RE_x[inds]

    # speed_threshold = 100
    # inds = (Results[n].speed < speed_threshold)
    # x = Results[n].LE_vx[inds]
    # y = Results[n].RE_vx[inds]

    sns.kdeplot(ax=axes[n], x=x, y=y, fill=True, cmap="Blues", levels=10, thresh=0.05)
    # axes[n].axis([-20, 20, -20, 20])  # [xmin, xmax, ymin, ymax]
    
    correlation, p_value = spearmanr(x, y)
    axes[n].text(0.05, 0.95, f"ρ = {correlation:.2f}\np = {p_value:.2g}", transform=axes[n].transAxes, verticalalignment='top')
    
    axes[n].set_title(f"Ferret {Results[n].id}")# | EO {Results[n].eo}")


 # %% head vs eye velocity scatter (trying to isolate horizontal)
n = 4
velocity_threshold = 25
inds = (np.abs(Results[n].LE_vx) > velocity_threshold) & (
    np.abs(Results[n].RE_vx) > velocity_threshold)
x = Results[n].yaw_v[~inds]
y = Results[n].RE_vx[~inds]
sns.scatterplot(x=x, y=y)



# %% ------------ BEGIN ADDING NEW ANALYSES HERE --------------


# %% MAIN SEQUENCE: amplitude vs peak velocity (log-log), eyes combined
pool_by_eo = False  # False: one panel per session | True: one panel per EO

if pool_by_eo:

    unique_eos = sorted(set(R.eo for R in Results))
    fig, axes = create_subplot_grid(len(unique_eos))
    groups = [[R for R in Results if R.eo == eo] for eo in unique_eos]
    titles = [f"EO {eo}" for eo in unique_eos]

else:

    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

for ax, group, title in zip(axes, groups, titles):

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group])
    pkv = np.concatenate([np.concatenate([R.df_LE["peak_velocity_deg_s"].to_numpy(),
                                          R.df_RE["peak_velocity_deg_s"].to_numpy()]) for R in group])

    inds = (amp > 0) & (pkv > 0)
    x = np.log10(amp[inds])
    y = np.log10(pkv[inds])

    if x.size:
        sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")


# %% MAIN SEQUENCE: amplitude vs duration (ms), eyes combined
pool_by_eo = False  # False: one panel per session | True: one panel per EO

if pool_by_eo:
    unique_eos = sorted(set(R.eo for R in Results))
    fig, axes = create_subplot_grid(len(unique_eos))
    groups = [[R for R in Results if R.eo == eo] for eo in unique_eos]
    titles = [f"EO {eo}" for eo in unique_eos]
else:
    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

for ax, group, title in zip(axes, groups, titles):

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group])
    dur = np.concatenate([np.concatenate([(R.df_LE["peak"] - R.df_LE["onset"]).to_numpy(),
                                          (R.df_RE["peak"] - R.df_RE["onset"]).to_numpy()]) for R in group])
    dur_ms = dur / 120.0 * 1000.0

    inds = (amp > 0) & np.isfinite(dur_ms)
    x = amp[inds]
    y = dur_ms[inds]

    if x.size:
        sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Duration (ms)")



# %% EYE-HEAD TIMING: cross-correlogram + coincidence
n = 0  # which session

fig = plot_eye_head_correlogram(Results[n], eye="RE")
# plt.savefig(f'{SAVELOC}/eye_head_timing_LE_{n}.svg', format='svg', bbox_inches='tight')

# raw numbers if wanted directly:
lags, counts_all, counts_same, counts_opp = eye_head_correlogram(Results[n], eye="RE", max_lag=120, bin_size=6)
coinc = eye_head_coincidence(Results[n], eye="RE", window=12, n_shuffles=1000)
print(f"head accompanied by eye: {coinc['frac_head_with_eye']:.2f}")
print(f"eye near head: {coinc['frac_eye_with_head']:.2f} (chance {coinc['chance_eye_with_head']:.2f})")




# %% GMM to look when running or not running 
# develop state machine in the future?
# combine speed and pupil size to look at different states of arousal and movement
# and how they change over development. 
# For example, are there more periods of high speed and large pupil size (active exploration) 
# in older animals compared to younger animals? 
# Are there more periods of low speed and small pupil size (quiescence) in younger animals compared to older animals? 
# This could be done with a Gaussian Mixture Model to identify clusters in the speed-pupil space, 
# and then look at the proportion of time spent in each cluster across ages.

from sklearn.mixture import GaussianMixture
from scipy.stats import norm

data = speeds[0]

# Fit the model
gmm = GaussianMixture(n_components=2, random_state=100)
gmm.fit(data.reshape(-1, 1))  # needs shape (n_samples, n_features)

# Extract the estimates
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_ 

x = np.linspace(data.min(), data.max(), 500)

# Plot histogram
plt.hist(data, bins=100, density=True, alpha=0.4, label='Data')

# Plot each component and the mixture
for i in range(2):
    component = weights[i] * norm.pdf(x, means[i], stds[i])
    plt.plot(x, component, label=f'Component {i+1} (μ={means[i]:.2f})')

# Total mixture
total = sum(weights[i] * norm.pdf(x, means[i], stds[i]) for i in range(2))
plt.plot(x, total, 'k--', label='Mixture')
plt.legend()
plt.show()

