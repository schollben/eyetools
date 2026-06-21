# NOTE: this is a scratch script to check the data and parameters for saccade detection. It is not meant to be run as a whole, but rather to be run in parts to check the data and parameters.
# must run the main script to load the data before running this script and import functions from the main script
#
# 
# 
# # %% CHECK eye saccades or gaze shifts
# do the threshold parameters make sense for this animal and/or session?
n = 0
df = extract_saccades(Results[n],
                      'eye', 
                      eye='LE', 
                      velocity_threshold=60,
                      min_duration=12,
                      min_inter_event=12)

y1 = Results[n].LE_x
# y1 = Results[n].LE_gaze_horizontal_deg

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



# %% check head saccades
n = 0
df = extract_saccades(Results[n],
                      'skull', 
                      velocity_threshold=2,
                      min_duration=6,
                      max_duration=600,
                      min_inter_event=6)


y1 = np.unwrap(Results[n].yaw)
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
