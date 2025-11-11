# %% init and load up data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import getEvents
from loadData import loadData
import plotStuff

# find/identify files to import. for now, hardcoded. 
# later make this a recursive loop that sifts through a folder structure 
# and gets data labels based on folder name
dataDir = '/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/VisBehavDev/data/'
paths = {}

paths['EO5'] = dataDir + 'session_2025-07-01_ferret_757_EyeCameras_P33_EO5/'
paths['EO15'] = dataDir + 'session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/'
paths['EO2'] = dataDir + 'session_2025-10-11_ferret_402_E02/' # NO TOY DATA
paths['EO10'] = dataDir + 'session_2025-10-20_ferret_420_E010/'

# outputs: xL, yL, xR, yR, xH, yH, yaw, pitch, roll, speed, heading, xToy, yToy
# paths - dict of data paths
# applyNaN - whether to replace eye data with NaNs based on confidence thresholds (i.e. remove blinks)
# verbose - whether to show timing alignment plots

results = loadData(paths, applyNaN=True, verbose=False)







# %% generate a comprehensive plot of all data for a given time window (in frames)
# make look better in seaborne?
importlib.reload(plotStuff)
# window - frame number range
ID = 'EO15'
start = 1000
dur = 30 # seconds
plotStuff.quickPlot(
    xPosLeye=results[ID].xL,
    yPosLeye=results[ID].yL,
    xPosReye=results[ID].xR,
    yPosReye=results[ID].yR,
    yaw=results[ID].yaw,
    pitch=results[ID].pitch,
    roll=results[ID].roll,
    speed=results[ID].speed,
    window=[start, start + 120*dur]
)


# %%

# %% DETECT SACCADES AND SACCADE-TRIGGERED AVERAGES AND MAKE PLOTS
#
# detect and plot EYE saccades from one eye position data (here, right eye horizontal)
# this is dumb method--not considering velocity
# might need to look at hand scoring or template matching? 
# what did Kerr/Neil do?
# note: 'events' are saccades per second (combining nasal and temporal) -> generate a distribution

importlib.reload(getEvents)

time_axis, nasal_STAs, temporal_STAs, head_STA, events = getEvents.detectSacs(
    posL=results['EO15'].xL,
    posR=results['EO15'].xR,
    yaw=results['EO15'].yaw,
    verbose=False
)

# list to arrays
nasal_STAs = np.array(nasal_STAs)
temporal_STAs = np.array(temporal_STAs)
head_STA = np.array(head_STA)

#  mean and SEM
mean_nasal = np.nanmean(nasal_STAs, axis=0)
sem_nasal = np.nanstd(nasal_STAs, axis=0) / np.sqrt(nasal_STAs.shape[0])
mean_temporal = np.nanmean(temporal_STAs, axis=0)
sem_temporal = np.nanstd(temporal_STAs, axis=0) / np.sqrt(temporal_STAs.shape[0])
mean_head = np.nanmean(head_STA, axis=0)
sem_head = np.nanstd(head_STA, axis=0) / np.sqrt(head_STA.shape[0])

plt.plot(time_axis, mean_nasal, 'b', label='Nasal')
plt.fill_between(time_axis,
                mean_nasal - sem_nasal,
                mean_nasal + sem_nasal,
                color='blue', alpha=0.2)

plt.plot(time_axis, mean_temporal, 'r', label='Temporal')
plt.fill_between(time_axis,
                mean_temporal - sem_temporal,
                mean_temporal + sem_temporal,
                color='red', alpha=0.2)

plt.plot(time_axis, mean_head, 'g', label='head')
plt.fill_between(time_axis,
                mean_head - sem_head,
                mean_head + sem_head,
                color='green', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()


# %% PLOT DISTRIBUTION OF SACCADES PER SECOND ACROSS TIMEPOINTS

plt.figure(figsize=(8,6))
bins = np.linspace(0, 10, 25)
for ID in ['EO2', 'EO5', 'EO10','EO15']:   
    time_axis, nasal_STAs, temporal_STAs, head_STA, events = getEvents.detectSacs(
        posL=results[ID].xL,
        posR=results[ID].xR,
        yaw=results[ID].yaw,
        verbose=False
    )
    plt.hist(events, bins=bins, alpha=0.5, label=ID, density=True)
plt.xlabel('Saccades per second')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.show()




# %% detect HEAD saccades from one eye position data (here, right eye horizontal)
# HOW TO DETECT HEAD SACCADES? NEED CIRCULAR DETECTION ALGO?
#note: adjust min_prominence to get reasonable number of events given head dynamics (need to make better)
#also note: the eye movements here are kinda shit b/c this isn't gaze-in-world but gaze-in-head (so still includes head VOR contributions, ect)
#are left/right flipped for EO2? maybe eye0 and eye1 cameras are swapped comapred to the other data

# time_axis, eye_STA, head_STA = getEvents.detectHeadSacs(
#     posL=results['EO15'].xL,
#     posR=results['EO15'].xR,
#     yaw=results['EO15'].yaw,
#     min_prominence=20
# )

time_axis, eye_STA, head_STA = getEvents.detectHeadSacs(
    posL=results['EO10'].xL,
    posR=results['EO10'].xR,
    yaw=results['EO10'].yaw,
    min_prominence=3
)

# time_axis, eye_STA, head_STA = getEvents.detectHeadSacs(
#     posL=results['EO5'].xL,
#     posR=results['EO5'].xR,
#     yaw=results['EO5'].yaw,
#     min_prominence=0.5
# )

# time_axis, eye_STA, head_STA = getEvents.detectHeadSacs(
#     posL=results['EO2'].xR,
#     posR=results['EO2'].xL,
#     yaw=results['EO2'].yaw,
#     min_prominence=0.5
# )


# list to arrays
eye_STA = np.array(eye_STA)
head_STA = np.array(head_STA)

#  mean and SEM
mean_eye = np.nanmean(eye_STA, axis=0)
sem_eye = np.nanstd(eye_STA, axis=0) / np.sqrt(eye_STA.shape[0])
mean_head = np.nanmean(head_STA, axis=0)
sem_head = np.nanstd(head_STA, axis=0) / np.sqrt(head_STA.shape[0])


fig, ax1 = plt.subplots(figsize=(8,4))

# Left axis: eye
ln1 = ax1.plot(time_axis, mean_eye, color='tab:blue', label='eye')
ax1.fill_between(time_axis,
                 mean_eye - sem_eye,
                 mean_eye + sem_eye,
                 color='tab:blue', alpha=0.2)
ax1.set_xlabel('Time (frames)')
ax1.set_ylabel('Eye (deg)', color='tab:blue')
ax1.tick_params(axis='y', colors='tab:blue')
ax1.set_ylim([-1, 5])

# Right axis: head
ax2 = ax1.twinx()
ln2 = ax2.plot(time_axis, mean_head, color='tab:green', label='head')
ax2.fill_between(time_axis,
                 mean_head - sem_head,
                 mean_head + sem_head,
                 color='tab:green', alpha=0.2)
ax2.set_ylabel('Head (deg)', color='tab:green')
ax2.tick_params(axis='y', colors='tab:green')
# ax2.set_ylim([0, 200])

plt.tight_layout()
plt.show()



# %% compare EO05 and EO15 speed distributions

bins = np.linspace(0, 100, 50)
plt.hist(results['EO5'].speed, bins=bins, alpha=0.8, label='Animal 1', color='orange', density=True)
plt.hist(results['EO15'].speed, bins=bins, alpha=0.8, label='Animal 2', color='blue', density=True)
plt.show()


# %% horizontal eye velocity 
# in degrees per second

bins = np.linspace(0, 600, 50)
d1 = np.concatenate((np.gradient(results['EO5'].xR,1/120),
                     np.gradient(results['EO5'].xL,1/120)))

plt.hist(d1, bins=bins, alpha=0.8, label='Animal 1', color='blue', density=True)

d2 = np.concatenate((np.gradient(results['EO15'].xR,1/120),
                     np.gradient(results['EO15'].xL,1/120)))

plt.hist(d2, bins=bins, alpha=0.8, label='Animal 2', color='orange', density=True)
plt.show()



# %% binocular correlation and running
# remember to flip one eye (choosing xR here) so EO15 shows binocular correlation
fig, axes = plt.subplots(1, 2, figsize=(8, 8))

speed = results['EO5'].speed
xR = -results['EO5'].xR
xL = results['EO5'].xL

inds = speed > -1 #Cm per second (choosing random value)

axes[0].scatter( xR[inds], xL[inds], 10, color='orange',alpha=.05)

speed = results['EO15'].speed
xR = -results['EO15'].xR
xL = results['EO15'].xL

inds = speed > -1 #Cm per second (choosing random value)

axes[1].scatter( xR[inds], xL[inds], 10, color='blue',alpha=.05)

for ax in axes:
     ax.set_xlim(-60,60)
     ax.set_xlabel('Right Eye Position (deg)')
     ax.set_ylabel('Left Eye Position (deg)')
     ax.set_ylim(-60,60)
     ax.set_aspect('equal')
     ax.tick_params(direction='out')
plt.tight_layout()


# %% trying a sliding window correlation

from scipy.stats import pearsonr

fs = 120               # Hz
window = 120*1        # 120 frames = 1 second

xL = np.copy(results['EO5'].xL)
xR = -np.copy(results['EO5'].xR)

# Sliding correlation
corrs_05 = []
idx_05 = []

for i in range(0, len(xR) - window + 1):
    xr_win = xR[i:i+window]
    xl_win = xL[i:i+window]
    mask = ~np.isnan(xr_win) & ~np.isnan(xl_win)
    if np.sum(mask) > 10:
        r, _ = pearsonr(xr_win[mask], xl_win[mask])
        corrs_05.append(r)
    else:
        corrs_05.append(np.nan)
    idx_05.append(i + window//2)

corrs_05 = np.array(corrs_05)
idx_05 = np.array(idx_05)


xL = np.copy(results['EO15'].xL)
xR = -np.copy(results['EO15'].xR)

corrs_15 = []
idx_15 = []

for i in range(0, len(xR) - window + 1):
    xr_win = xR[i:i+window]
    xl_win = xL[i:i+window]
    mask = ~np.isnan(xr_win) & ~np.isnan(xl_win)
    if np.sum(mask) > 10:
        r, _ = pearsonr(xr_win[mask], xl_win[mask])
        corrs_15.append(r)
    else:
        corrs_15.append(np.nan)
    idx_15.append(i + window//2)

corrs_15 = np.array(corrs_15)
idx_15 = np.array(idx_15)

bins = np.linspace(-1, 1, 25)
plt.hist(corrs_05, bins=bins, alpha=0.8, label='Animal 1', color='orange', density=True)
plt.hist(corrs_15, bins=bins, alpha=0.8, label='Animal 2', color='blue', density=True)
plt.show()



# %% EYE MOVEMENTS RELATIVE TO TOY POSITION AND RUNNING

fig, axes = plt.subplots(1, 3, figsize=(8, 8))

xL = results['EO5'].xL
yL = results['EO5'].yL
xR = results['EO5'].xR
yR = results['EO5'].yR

# distance of toy and head
dist = np.sqrt( (results['EO5'].xH - results['EO5'].xToy)**2 +
                (results['EO5'].yH - results['EO5'].yToy)**2 )
speed = results['EO5'].speed
inds = (dist < 40) & (speed > 5) #Cm per second (choosing random value)


# xL = results['EO15'].xL
# yL = results['EO15'].yL
# xR = results['EO15'].xR
# yR = results['EO15'].yR

# # distance of toy and head
# dist = np.sqrt( (results['EO15'].xH - results['EO15'].xToy)**2 +
#                 (results['EO15'].yH - results['EO15'].yToy)**2 )
# speed = results['EO15'].speed
# inds = (dist < 20) & (speed > 5) #Cm per second (choosing random value)



axes[0].scatter( xR, yR, 5, color='gray',alpha=.025)
axes[0].scatter(xR[inds], yR[inds], s=5, facecolors='red', edgecolors='white', linewidths=0.5, alpha=0.1)
axes[0].set_title('Right Eye')

axes[1].scatter( xL, yL, 5, color='gray',alpha=.025)
axes[1].scatter(xL[inds], yL[inds], s=5, facecolors='red', edgecolors='white', linewidths=0.5, alpha=0.1)
axes[1].set_title('Left Eye')

axes[2].scatter( -xL, xR, 5, color='gray',alpha=.025)
axes[2].scatter(-xL[inds], xR[inds], s=5, facecolors='red', edgecolors='white', linewidths=0.5, alpha=0.1)
axes[2].set_title('Binocular Horizontal Movements')

for ax in axes:
     ax.set_xlim(-60,60)
     ax.set_xlabel('Right Eye Position (deg)')
     ax.set_ylabel('Left Eye Position (deg)')
     ax.set_ylim(-60,60)
     ax.set_aspect('equal')
     ax.tick_params(direction='out')
plt.tight_layout()
