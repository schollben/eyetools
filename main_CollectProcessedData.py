# %% main script to run data loading, cleaning, and saccade extraction for a session

# main init
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_session_data, removeBadData, extract_saccades, saccade_triggered_average, saccade_andHead_triggered_average
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# plotting setup
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'
saveloc = "/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/VisBehavDev/rawfigures/"

from data_viewer import launch_viewer
# to check the data and extracted saccades for a session, use:
# launch_viewer(R)

# load data
SESSION = [
"session_2025-07-07_ferret_757_EyeCameras_P39_E11_analyzable_output",
"session_2025-07-05_ferret_757_EyeCameras_P37_EO9_analyzable_output",
"session_2025-07-01_ferret_757_EyeCameras_P33_EO5_analyzable_output",
"session_2025-06-29_ferret_757_EyeCameras_P31_EO3__1_analyzable_output",
"session_2025-06-28_ferret_757_EyeCameras_P30_EO2_analyzable_output",
]

Results = []
for session in SESSION:
    R = load_session_data(session)
    removeBadData(R)
    Results.append(R)


# %% examing eye kinematic changes over development
from scipy.signal import decimate

eos = []
eyeRates = []
speeds = []
gazeRate = []
angVelocities = []

for n in range(len(Results)):

    df_LE = extract_saccades(Results[n], 'eye', eye='LE',
                             velocity_threshold=40, min_duration=12, min_inter_event=12)

    df_head = extract_saccades(Results[n],'skull', 
                        velocity_threshold=2, min_duration=12, max_duration=600, min_inter_event=12)

    df_LEgaze = extract_saccades(Results[n], 'gaze', eye='LE',
                             velocity_threshold=2, min_duration=12, min_inter_event=12)

    #sliding window to calculate saccade rate
    #minimum amplitude saccade?
    window_len = 120 * 10
    n_frames = len(Results[n].EQframes)
    starts = np.arange(0, n_frames - window_len + 1, window_len // 2)

    eos.append(Results[n].eo)

    rate = np.array([(1/10) * np.sum((df_LE.onset >= s) & (df_LE.onset < s + window_len)) for s in starts])
    eyeRates.append(rate)

    vv = np.sqrt( Results[n].linearVel_x ** 2 + Results[n].linearVel_y ** 2 ).astype(int)
    
    angVel =  np.sqrt(Results[n].roll_v ** 2 + Results[n].pitch_v ** 2 + Results[n].yaw_v ** 2 ).astype(int) #total angular velocity -> use to examine stablization?
    angVel = angVel[(vv >= 0) & (vv < 800)]
    angVelocities.append(angVel)

    vv = vv[(vv >= 0) & (vv < 800)] # need to go understand why there are negative values in the linear velocity, which should be absolute value of speed. For now, just remove them.
    speeds.append(vv)

    rate = np.array([(1/10) * np.sum((df_LEgaze.onset >= s) & (df_LEgaze.onset < s + window_len)) for s in starts])
    gazeRate.append(rate)
    
    fig = saccade_triggered_average(Results[n], df_LE, window=60)
    plt.savefig(f'{saveloc}/LE_pos_eo_{n}.svg', format='svg', bbox_inches='tight')

    fig = saccade_andHead_triggered_average(Results[n], df_head, window=120)
    plt.savefig(f'{saveloc}/head_and_LE_pos_eo_{n}.svg', format='svg', bbox_inches='tight')



# %% compare distributions of metrics amplitudes between 2 ages

fig, axes = plt.subplots(1, 4, figsize=(4, 1), sharex=False)
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


plt.savefig(f'{saveloc}/histPlots.svg', format='svg', bbox_inches='tight')




# %% check head saccades

df = extract_saccades(Results[0],
                      'skull', 
                      velocity_threshold=2,
                      min_duration=6,
                      max_duration=600,
                      min_inter_event=6)


y1 = np.unwrap(Results[0].yaw)
y2 = np.full(len(y1), np.nan)
y3 = np.full(len(y1), np.nan)

for j in range(len(df)):
        y2[df['onset'][j]] = y1[df['onset'][j]]
        y3[df['peak'][j]] = y1[df['peak'][j]]

win = np.arange(15 * 1e3, 25 * 1e3).astype(int)
y1 = y1[win]
y2 = y2[win]
y3 = y3[win]

fig = go.Figure()
fig.add_trace(go.Scatter(x=win , y=y1))
fig.add_trace(go.Scatter(x=win , y=y2, mode='markers', marker=dict(color='orange', size=6)))
fig.add_trace(go.Scatter(x=win , y=y3, mode='markers', marker=dict(color='red', size=6)))



# %% check eye saccades or gaze shifts

df = extract_saccades(Results[3],
                      'eye', 
                      eye='LE', 
                      velocity_threshold=60,
                      min_duration=12,
                      min_inter_event=12)

y1 = Results[3].LE_x
# y1 = Results[3].LE_gaze_horizontal_deg

y2 = np.full(len(y1), np.nan)
y3 = np.full(len(y1), np.nan)

for j in range(len(df)):
        y2[df['onset'][j]] = y1[df['onset'][j]]
        y3[df['peak'][j]] = y1[df['peak'][j]]

win = np.arange(20 * 1e3, 25 * 1e3).astype(int)
y1 = y1[win]
y2 = y2[win]
y3 = y3[win]

fig = go.Figure()
fig.add_trace(go.Scatter(x=win , y=y1))
fig.add_trace(go.Scatter(x=win , y=y2, mode='markers', marker=dict(color='orange', size=6)))
fig.add_trace(go.Scatter(x=win , y=y3, mode='markers', marker=dict(color='red', size=6)))


