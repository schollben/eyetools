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

from data_viewer import launch_viewer
# to check the data and extracted saccades for a session, use:
# launch_viewer(R)

# load data
SESSION = [
"session_2025-07-07_ferret_757_EyeCameras_P39_E11_analyzable_output",
"session_2025-07-05_ferret_757_EyeCameras_P37_EO9_analyzable_output",
"session_2025-07-01_ferret_757_EyeCameras_P33_EO5_analyzable_output",
"session_2025-06-29_ferret_757_EyeCameras_P31_EO3__1_analyzable_output",
# "session_2025-06-28_ferret_757_EyeCameras_P30_EO2_analyzable_output",
]

Results = []
for session in SESSION:
    R = load_session_data(session)
    removeBadData(R)
    Results.append(R)


# %% examing eye kinematic changes over development

eos, pupil_m, pupil_s, rate_m, rate_s, amp_m, amp_s, ampVel_m, ampVel_s  = [], [], [], [], [], [], [], [], []

for n in range(len(Results)):

    df_LE = extract_saccades(Results[n], 'eye', eye='LE',
                             velocity_threshold=60, min_duration=30, min_inter_event=12)

    df_head = extract_saccades(Results[n],'skull', 
                        velocity_threshold=.11, min_duration=12, max_duration=600, min_inter_event=12)

    #sliding window to calculate saccade rate
    #minimum amplitude saccade?
    window_len = 120 * 10
    n_frames = len(Results[n].EQframes)
    starts = np.arange(0, n_frames - window_len + 1, window_len // 2)
    rate = np.array([(1/10) * np.sum((df_LE.onset >= s) & (df_LE.onset < s + window_len)) for s in starts])

    eos.append(Results[n].eo)

    pupil_m.append(np.nanmean( Results[n].LE_pupil[ Results[n].LE_pupil < 10 ] )) #ignore outliers 
    pupil_s.append(np.nanstd( Results[n].LE_pupil[ Results[n].LE_pupil < 10 ] ))

    rate_m.append(np.nanmean(rate))
    rate_s.append(np.nanstd(rate))

    amp_m.append(np.nanmean(df_LE.amplitude_deg))
    amp_s.append(np.nanstd(df_LE.amplitude_deg))

    ampVel_m.append(np.nanmean(df_LE.peak_velocity_deg_s))
    ampVel_s.append(np.nanstd(df_LE.peak_velocity_deg_s))

    fig = saccade_triggered_average(Results[n], df_LE, window=60)
    fig = saccade_andHead_triggered_average(Results[n], df_head, window=60)


# %%

fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharex=False)
fig.tight_layout(w_pad=3) 
sns.despine(fig=fig)

axes[0, 0].errorbar(
    eos, pupil_m, yerr=pupil_s,
    fmt='o',
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=0.5, 
    ecolor="black",
    capsize=0,                  
    capthick=1,
)

axes[0, 1].errorbar(
    eos, rate_m, yerr=rate_s,
    fmt='o',                   
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=0.5, 
    ecolor="black",
    capsize=0,                  
    capthick=1,
)

axes[1, 0].errorbar(
    eos, amp_m, yerr=amp_s,
    fmt='o',                   
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=0.5,               
    ecolor="black",
    capsize=0,                  
    capthick=1,
)

axes[1, 1].errorbar(
    eos, ampVel_m, yerr=ampVel_s,
    fmt='o',                   
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=0.5,               
    ecolor="black",
    capsize=0,                  
    capthick=1,
)




# %%

eos, rate_gazem, rate_gazes, rate_headm, rate_heads = [], [], [], [], []
for n in range(len(Results)):

    df_LEgaze = extract_saccades(Results[n], 'gaze', eye='LE',
                             velocity_threshold=20, min_duration=12, min_inter_event=12)

    df_head = extract_saccades(Results[n],'skull', 
                        velocity_threshold=2, min_duration=6, max_duration=600, min_inter_event=6)

    #sliding window to calculate saccade rate
    #minimum amplitude saccade?
    window_len = 120 * 10
    n_frames = len(Results[n].EQframes)
    starts = np.arange(0, n_frames - window_len + 1, window_len // 2)
    rate_gaze = np.array([(1/10) * np.sum((df_LEgaze.onset >= s) & (df_LEgaze.onset < s + window_len)) for s in starts])
    rate_head = np.array([(1/10) * np.sum((df_head.onset >= s) & (df_head.onset < s + window_len)) for s in starts])

    eos.append(Results[n].eo)

    rate_gazem.append(np.nanmean(rate_gaze))
    rate_gazes.append(np.nanstd(rate_gaze))
    rate_headm.append(np.nanmean(rate_head))
    rate_heads.append(np.nanstd(rate_head))

axes[1, 0].errorbar(
    eos, rate_headm, yerr=rate_heads,
    fmt='o',                   
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=1,               
    capsize=0,                  
    capthick=1,
    title="head rate"                 
)

axes[1, 1].errorbar(
    eos, rate_gazem, yerr=rate_gazes,
    fmt='o',                   
    color="#72B3E9",            
    markeredgecolor='black',    
    markeredgewidth=0.5,
    elinewidth=1,               
    capsize=0,                  
    capthick=1
    )




## % compare values from 2 ages for a given metric, e.g. pupil size distribution

fig, ax = plt.subplots(figsize=(4, 3))
v1 = Results[0].LE_pupil
v1 = v1[v1 < 10] #ignore outliers
v2 = Results[4].LE_pupil
v2 = v2[v2 < 10] #ignore outliers
sns.kdeplot(v1, ax=ax, color='blue', fill=True, alpha=0.3)
sns.kdeplot(v2, ax=ax, color='red',  fill=True, alpha=0.3)


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


