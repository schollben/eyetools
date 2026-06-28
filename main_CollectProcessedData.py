# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init

# %load_ext autoreload
# %autoreload 2
import sys
# Set your paths in local_config.py (copy local_config.py.example to get started).
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import load_session_data, removeBadData, extract_saccades, get_sessions_by_ferret, get_sessions
from utils import saccade_triggered_average, saccade_triggered_average, saccade_andHead_triggered_average
from utils.config import SAVELOC
import plotly.graph_objects as go
import numpy as np
from data_viewer import launch_viewer
# plotting setup
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
    Results.append(R)
n_sesh = len(Results)
print(n_sesh, "sessions loaded")

# to look at data execute: launch_viewer(load_session_data(SESSION[n]))
# or launch_viewer(R) if there is only 1 session in the list


# %% collect data on eye kinematic changes over development (added to Results object)
eos = []
eyeRatesLE = []
eyeRatesRE = []
speeds = []
gazeRateLE = []
gazeRateRE = []
angVelocities = []
pupilSizes = []

for n in range(len(Results)):

    df_LE = extract_saccades(Results[n], 'eye', eye='LE',
                             velocity_threshold=40, min_duration=12, min_inter_event=12)

    df_RE = extract_saccades(Results[n], 'eye', eye='RE',
                             velocity_threshold=40, min_duration=12, min_inter_event=12)

    df_head = extract_saccades(Results[n],'skull', 
                        velocity_threshold=2, min_duration=12, max_duration=600, min_inter_event=12)

    df_LEgaze = extract_saccades(Results[n], 'gaze', eye='LE',
                             velocity_threshold=2, min_duration=12, min_inter_event=12)
    
    df_REgaze = extract_saccades(Results[n], 'gaze', eye='RE',
                             velocity_threshold=2, min_duration=12, min_inter_event=12)

    #sliding window to calculate saccade rate
    #minimum amplitude saccade?
    window_len = 120 * 5
    n_frames = len(Results[n].EQframes)
    starts = np.arange(0, n_frames - window_len + 1, window_len // 2)

    eos.append(Results[n].eo)

    rate = np.array([(1/10) * np.sum((df_LE.onset >= s) & (df_LE.onset < s + window_len)) for s in starts])
    eyeRatesLE.append(rate)

    rate = np.array([(1/10) * np.sum((df_RE.onset >= s) & (df_RE.onset < s + window_len)) for s in starts])
    eyeRatesRE.append(rate)

    vv = np.sqrt( Results[n].linearVel_x ** 2 + Results[n].linearVel_y ** 2 )
    indFrames = (vv >= 0) & (vv < 800) # need to go understand why there are negative values in the linear velocity, which should be absolute value of speed. For now, just remove them.

    # nan out values above 800 mm/s, which are likely artifacts. Need to check the raw data to understand why these values are present and if there is a way to clean them up without just removing them.

    angVel =  np.sqrt(Results[n].roll_v ** 2 + Results[n].pitch_v ** 2 + Results[n].yaw_v ** 2 ) #total angular velocity -> use to examine stablization?
    angVelocities.append(angVel[indFrames])

    pupilSizes.append(Results[n].LE_pupil[indFrames]) # examine pupil size changes over development, as a proxy for arousal or cognitive effort. Need to check if there are any differences in the eye tracking quality that could affect this metric.

    speeds.append(vv[indFrames])

    rate = np.array([(1/10) * np.sum((df_LEgaze.onset >= s) & (df_LEgaze.onset < s + window_len)) for s in starts])
    gazeRateLE.append(rate)

    rate = np.array([(1/10) * np.sum((df_REgaze.onset >= s) & (df_REgaze.onset < s + window_len)) for s in starts])
    gazeRateRE.append(rate)
    
    fig = saccade_triggered_average(Results[n], 
                                    df_LE, df_RE, 
                                    window=30, 
                                    binocular="combined")
    if SAVELOC:
        plt.savefig(f'{SAVELOC}/LE_pos_eo_{n}.svg', format='svg', bbox_inches='tight')
    
    fig = saccade_andHead_triggered_average(Results[n], 
                                            df_head, 
                                            df_LE, df_RE, 
                                            window=120, 
                                            binocular="combined")
    # plt.savefig(f'{saveloc}/head_and_LE_pos_eo_{n}.svg', format='svg', bbox_inches='tight')


# %% BASIC PLOTS: pupil distribution
n = 0 # need to specify which session to plot (n = 0 for a single session, or n = 0,1,2,3 for multiple sessions)
pL = Results[n].LE_pupil[ Results[n].LE_pupil < 10 ]
pR = Results[n].RE_pupil[ Results[n].RE_pupil < 10 ]

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
sns.despine(fig=fig)
sns.histplot(pL, ax=axes, color="#725EE7", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
sns.histplot(pR, ax=axes, color="#E93115", stat="probability", binwidth=0.04, linewidth=0, alpha=0.6)
axes.set_xlabel("Pupil size (mm)")


# %% BASIC PLOTS: saccade rate distribution
n = 0 # need to specify which session to plot (n = 0 for a single session, or n = 0,1,2,3 for multiple sessions)
pL = eyeRatesLE[n]
pR = eyeRatesRE[n]

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
sns.despine(fig=fig)
sns.histplot(pL, ax=axes, color="#725EE7", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
sns.histplot(pR, ax=axes, color="#E93115", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
axes.set_xlabel("Saccade rate (Hz)")


# %% BASIC PLOTS: gaze rate distribution
n = 0 # need to specify which session to plot (n = 0 for a single session, or n = 0,1,2,3 for multiple sessions)
pL = gazeRateLE[n]
pR = gazeRateRE[n]

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
sns.despine(fig=fig)
sns.histplot(pL, ax=axes, color="#725EE7", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
sns.histplot(pR, ax=axes, color="#E93115", stat="probability", binwidth=0.1, linewidth=0, alpha=0.6)
axes.set_xlabel("Gaze rate (Hz)")


# %% BASIC PLOTS: speed distribution
n = 0 
dat = speeds[n]

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
sns.despine(fig=fig)
sns.histplot(dat, ax=axes, color="#000000", stat="probability", binwidth=20, linewidth=0, alpha=0.6)
axes.set_xlabel("speed (mm/s")



# %% BASIC PLOT: speed vs angular velocity of head rotations

n = 0
x = speeds[n]
y = np.abs(angVelocities[n])
sns.scatterplot(x=x, y=y)

# %% 2D scatters comparing speed and pupil size distributions between ages (young to old)

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=False)
fig.tight_layout(w_pad=2)
n = 0
sns.kdeplot(ax=axes, x=speeds[n],y=pupilSizes[n], fill=True, cmap="Blues", levels=10, thresh=0.05)
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