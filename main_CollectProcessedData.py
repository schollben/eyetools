# %% main script to run data loading, cleaning, and saccade extraction for a session. This is a scratch space to test out the functions in utils.
# init
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_session_data, removeBadData, extract_saccades
import plotly.graph_objects as go
import numpy as np
# plotting setup
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
mpl.rcParams.update({
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'axes.grid':          False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial'],
    'font.size':          6,
    'svg.fonttype':       'none',
})

from data_viewer import launch_viewer
#use launch_viewer(R) to check the data and extracted saccades for a session.

# %% load data
SESSION = [
"session_2025-07-07_ferret_757_EyeCameras_P39_E11_analyzable_output",
"session_2025-07-05_ferret_757_EyeCameras_P37_EO9_analyzable_output",
"session_2025-07-01_ferret_757_EyeCameras_P33_EO5_analyzable_output",
"session_2025-06-29_ferret_757_EyeCameras_P31_EO3__1_analyzable_output",
"session_2025-06-28_ferret_757_EyeCameras_P30_EO2_analyzable_output"
]

Results = []
for session in SESSION:

    R = load_session_data(session)
    removeBadData(R)
    Results.append(R)


# %%

fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharex=True)

for n in range(len(Results)):

    eo = Results[n].eo

    df_LE = extract_saccades(Results[n],
                        'eye', 
                        eye='LE', 
                        velocity_threshold=60,
                        min_duration=12,
                        min_inter_event=3)
   
    df_RE = extract_saccades(Results[n],
                        'eye', 
                        eye='RE', 
                        velocity_threshold=60,
                        min_duration=12,
                        min_inter_event=3)

    pupil_m = np.nanmean(Results[n].LE_pupil)
    pupil_s = np.nanstd(Results[n].LE_pupil)
    Nsaccades = len(df_LE) / len(Results[n].EQframes)
    #compute in time windows over trace?
    
    axes[0].errorbar(eo,pupil_m,yerr=pupil_s, 
                    fmt='o-', capsize=4,
                    color='blue', markerfacecolor='blue',
                    markeredgecolor='white', ecolor='blue')

    axes[1].errorbar(eo ,Nsaccades,
                fmt='o-', capsize=4,
                color='blue', markerfacecolor='blue',
                markeredgecolor='white', ecolor='blue')

                    

    pupil_m = np.nanmean(Results[n].RE_pupil)
    pupil_s = np.nanstd(Results[n].RE_pupil)
    Nsaccades = len(df_RE) / len(Results[n].EQframes)

    axes[0].errorbar(eo,pupil_m,yerr=pupil_s, 
                    fmt='o-', capsize=4,
                    color='red', markerfacecolor='red',
                    markeredgecolor='white', ecolor='red')

    axes[1].errorbar(eo,Nsaccades,
                fmt='o-', capsize=4,
                color='red', markerfacecolor='red',
                markeredgecolor='white', ecolor='red')


    # %%

    # df_LEgaze = extract_saccades(Results,
    #                     'gaze', 
    #                     eye='LE', 
    #                     velocity_threshold=20,
    #                     min_duration=6,
    #                     min_inter_event=6)

    # df_REgaze = extract_saccades(Results,
    #                     'gaze', 
    #                     eye='RE', 
    #                     velocity_threshold=20,
    #                     min_duration=6,
    #                     min_inter_event=6)

    # df_head = extract_saccades(Results,
    #                     'skull', 
    #                     velocity_threshold=2,
    #                     min_duration=6,
    #                     max_duration=600,
    #                     min_inter_event=6)







# %% check head saccades 

df = extract_saccades(Results,
                      'skull', 
                      velocity_threshold=2,
                      min_duration=6,
                      max_duration=600,
                      min_inter_event=6)


y1 = np.unwrap(Results.yaw)
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

df = extract_saccades(Results,
                      'gaze', 
                      eye='LE', 
                      velocity_threshold=20,
                      min_duration=6,
                      min_inter_event=6)

# y1 = Results.LE_x
y1 = Results.LE_gaze_horizontal_deg

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


