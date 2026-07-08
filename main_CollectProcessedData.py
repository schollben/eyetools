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
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, get_sessions_by_ferret, get_sessions 
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
#SESSION = get_sessions_by_ferret(402, 420)    # multiple — preserves order by ferret
SESSION = get_sessions_by_ferret(420)          # or load sessions from an inidividual ID
#SESSION = get_sessions("session_2025-07-09_ferret_757_EyeCameras_P41_E13_analyzable_output") # or load a specific session by name
# SESSION = get_sessions("session_2026-03-16_ferret_403_P49_E7_analyzable_output") # or load a specific session by name

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
    axes[n].axis([1.5, 4.5, 0, 0.2])  # [xmin, xmax, ymin, ymax]
    axes[n].set_xticks([2, 3, 4])
    axes[n].set_yticks([0, 0.2])


# %% BASIC PLOTS: saccade rate distribution
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    pL = Results[n].eyeRateLE
    pR = Results[n].eyeRateRE

    sns.histplot(pL, ax=axes[n], color="#725EE7", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    sns.histplot(pR, ax=axes[n], color="#E93115", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    axes[n].axis([0, 4, 0, 0.5])  # [xmin, xmax, ymin, ymax]


# %% BASIC PLOTS: gaze rate distribution
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    pL = Results[n].gazeRateLE
    pR = Results[n].gazeRateRE

    sns.histplot(pL, ax=axes[n], color="#725EE7", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    sns.histplot(pR, ax=axes[n], color="#E93115", stat="probability", binwidth=0.25, linewidth=0, alpha=0.6)
    axes[n].axis([0, 4, 0, 0.5])  # [xmin, xmax, ymin, ymax]


# %% BASIC PLOTS: speed distribution (mm/s)
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    dat = Results[n].speed
    sns.histplot(dat, ax=axes[n], color="#000000", stat="probability", binwidth=20, linewidth=0, alpha=0.6)
    # axes[n].set_xscale("log") #need to remove 0s first


# %% BASIC PLOT: speed vs angular velocity of head rotations (all 3 axes    )
fig, axes = create_subplot_grid(n_sesh)

for n in range(n_sesh):

    inds = (Results[n].speed > 0) & (np.abs( Results[n].angVelocities) > 0)
    x = Results[n].speed[inds]
    y = np.abs( Results[n].angVelocities[inds])
    x = np.log10(x)
    y = np.log10(y)
    sns.scatterplot(ax=axes[n], x=x, y=y, s=1, alpha=0.1)
    # axes[n].axis([1, 800, 0, 50])  # [xmin, xmax, ymin, ymax]



# %% 2D scatters comparing speed and pupil size distributions between ages (young to old)

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
n = 0
sns.kdeplot(ax=axes, x=Results[n].speed, y=Results[n].pupilSize, fill=True, cmap="Blues", levels=10, thresh=0.05)
ax=axes.set_xlabel("speed (mm/s)")
ax=axes.set_ylabel("pupil size (mm)")


 # %% examine binocular coordination of eye movements
m = int(np.ceil(n_sesh/2))
fig, axes = plt.subplots(2, m, 
                         figsize=(2*m, 4), 
                         sharex=False, 
                         squeeze=False)
axes = axes.flatten()
fig.tight_layout(w_pad=2)

for n in range(n_sesh):

    # velocity_threshold = 0
    # inds = (Results[n].LE_vx > velocity_threshold) & (Results[n].RE_vx > velocity_threshold)
    # x = Results[n].LE_x[inds]
    # y = -Results[n].RE_x[inds]

    # speed_threshold = 0
    # inds = (speeds[n] > speed_threshold)
    x = Results[n].LE_vx
    y = -Results[n].RE_vx

    sns.kdeplot(ax=axes[n], x=x, y=y, fill=True, cmap="Blues", levels=10, thresh=0.05)
    axes[n].axis([-20, 20, -20, 20])  # [xmin, xmax, ymin, ymax]
    


 # %%
n = 4
velocity_threshold = 25
inds = (np.abs(Results[n].LE_vx) > velocity_threshold) & (
    np.abs(Results[n].RE_vx) > velocity_threshold)
x = Results[n].yaw_v[~inds]
y = Results[n].RE_vx[~inds]
sns.scatterplot(x=x, y=y)


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





# %% COMPARE distributions of metrics amplitudes between 2 ages

fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharex=False)
fig.tight_layout(w_pad=2)
sns.despine(fig=fig)

n = 3
p1 = Results[n].LE_pupil[ Results[n].LE_pupil < 10 ]
n = 1
p2 = Results[n].LE_pupil[ Results[n].LE_pupil < 10 ]
sns.histplot(p1, ax=axes[0], color="#848484", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
sns.histplot(p2, ax=axes[0], color="#000000", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
axes[0].set_xlabel("Pupil size (mm)")
axes[0].set_xlim(1, 4)
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_ylim(0, 0.12)
axes[0].set_yticks([0, 0.06, 0.12])

n = 3
sns.histplot(speeds[n], ax=axes[1], color="#848484", stat="probability", binwidth=20, linewidth=0, alpha=0.6)
n = 1
sns.histplot(speeds[n], ax=axes[1], color="#000000", stat="probability", binwidth=20, linewidth=0, alpha=0.6)
axes[1].set_title("")
axes[1].set_xlabel("speed (mm/s)")
axes[1].set_xlim(0, 600)
axes[1].set_xticks([0,300,600])
axes[1].set_ylim(0, 0.2)

n = 3
sns.histplot(eyeRates[n], ax=axes[2], color="#848484", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
n = 1
sns.histplot(eyeRates[n], ax=axes[2], color="#000000", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
axes[2].set_xlabel("Saccade rate (Hz)")
axes[2].set_xlim(0, 5)
axes[2].set_ylim(0, 0.3)

n = 3
sns.histplot(gazeRate[n], ax=axes[3], color="#848484", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
n = 1
sns.histplot(gazeRate[n], ax=axes[3], color="#000000", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
axes[3].set_xlabel("Gaze rate (Hz)")
axes[3].set_xlim(0, 5)
axes[3].set_ylim(0, 0.2)

# plt.savefig(f'{saveloc}/histPlots.svg', format='svg', bbox_inches='tight')