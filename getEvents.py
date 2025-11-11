# detect saccades from x or y position data
# identify nasal and temporal saccades (positive and negative velocity deflections)
# compute saccade triggered average of eye position data
# NEED TO COMBINE BOTH EYES

import numpy as np
from scipy.signal import find_peaks, medfilt, resample
from matplotlib import pyplot as plt

def detectSacs(posL,posR,yaw,verbose=True):

    # Parameters
    min_peak_height = 2
    min_peak_distance = int(10) 
    min_prominence = 10

    peaks_pos, _ = find_peaks(
        posL,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)

    peaks_neg, _ = find_peaks(
        -posL,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)

    peaks_pos2, _ = find_peaks(
        posR,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)

    peaks_neg2, _ = find_peaks(
        -posR,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)

    events = np.zeros((len(posL),1))
    events[peaks_pos] = 1
    events[peaks_neg] = 1
    print(events.sum(axis=0), "saccades detected (pos & neg)")
    events = events.ravel()  # make 1D
    events = resample(events, int(len(events)/120)) * 120 #saccades per second

    valid_neg = []
    for idx in peaks_neg:
        v = -posL[idx] - -posL[idx-3]
        if v > min_peak_height:
            valid_neg.append(idx)
    valid_neg = np.array(valid_neg, dtype=int)

    valid_pos = []
    for idx in peaks_pos:
        v = posL[idx] - posL[idx-3]
        if v > min_peak_height:
            valid_pos.append(idx)
    valid_pos = np.array(valid_pos, dtype=int)
    
    valid_neg2 = []
    for idx in peaks_neg2:
        v = -posR[idx] - -posR[idx-3]
        if v > min_peak_height:
            valid_neg2.append(idx)
    valid_neg2 = np.array(valid_neg2, dtype=int)

    valid_pos2 = []
    for idx in peaks_pos2:
        v = posR[idx] - posR[idx-3]
        if v > min_peak_height:
            valid_pos2.append(idx)
    valid_pos2 = np.array(valid_pos2, dtype=int)

    #check results?
    if verbose:
        t = np.arange(len(posL)) / 120
        plt.plot(t, posL, '-')
        plt.plot(t[valid_pos], posL[valid_pos], 'go')
        plt.plot(t[valid_neg], posL[valid_neg], 'yo')
        plt.xlim([0,10])
        plt.show()

    # saccade and head triggered average
    pre_time = 5  # frames
    post_time = 25  # frames
    time_axis = np.arange(-pre_time, post_time)
    nasal_STAs = []
    temporal_STAs = []
    head_STA = []

    for t in valid_pos:

        sac = posL[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:2])
            
        hdm = np.abs( yaw[t-pre_time : t+post_time] )
        hdm = hdm- np.nanmean(hdm[0:2])

        if t+post_time < len(posL):
                nasal_STAs.append(sac)
                head_STA.append(hdm)

    for t in valid_pos2:

        sac2 = posR[t-pre_time : t+post_time]
        sac2 = sac2 - np.nanmean(sac2[0:2])

        if t+post_time < len(posL):
             nasal_STAs.append(sac2)

    for t in valid_neg:

        sac = posL[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:2])
        
        sac2 = posR[t-pre_time : t+post_time]
        sac2 = sac2 - np.nanmean(sac2[0:2])

        hdm = np.abs( yaw[t-pre_time : t+post_time] )
        hdm = hdm- np.nanmean(hdm[0:2])

        if t+post_time < len(posL):
                temporal_STAs.append(sac)
                head_STA.append(hdm)

    for t in valid_neg2:

        sac2 = posR[t-pre_time : t+post_time]
        sac2 = sac2 - np.nanmean(sac2[0:2])

        if t+post_time < len(posL):
             temporal_STAs.append(sac2)

    return time_axis, nasal_STAs, temporal_STAs, head_STA, events





def detectHeadSacs(posL, posR, yaw, min_prominence=20):

    # Similar implementation as detectSacs but for head position data
    # Parameters
    min_peak_height = 0.1
    min_peak_distance = 30
    # min_prominence = 20

    hdChange = np.diff(medfilt(yaw,31))

    peaks_pos, _ = find_peaks(
        hdChange,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)
    
    # remove peaks where yaw is >160 or <20 degrees
    if len(peaks_pos) > 0:
        mask = (yaw[peaks_pos] <= 120) & (yaw[peaks_pos] >= 20)
        peaks_pos = peaks_pos[mask]

    peaks_neg, _ = find_peaks(
        -hdChange,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=min_prominence)
    
    # remove peaks where yaw is >160 or <20 degrees
    if len(peaks_neg) > 0:
        mask = (yaw[peaks_neg] <= 160) & (yaw[peaks_neg] >= 20)
        peaks_neg = peaks_neg[mask]

    print(len(peaks_pos) + len(peaks_neg), "head saccades detected (pos & neg)")
    
    # saccade and head triggered average
    pre_time = 10  # frames
    post_time = 20  # frames
    time_axis = np.arange(-pre_time, post_time)
    eye_STA = []
    head_STA = []

    # peaks_pos = peaks_pos[peaks_pos+post_time < len(posL)]
    for t in peaks_pos:
        
        sac = -posL[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:pre_time//2])
        if (np.isnan(sac).sum() < post_time) & (len(sac)==pre_time+post_time):
            eye_STA.append(sac)

        sac = posR[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:pre_time//2])
        if (np.isnan(sac).sum() < post_time) & (len(sac)==pre_time+post_time):
            eye_STA.append(sac)

        hdm = yaw[t-pre_time : t+post_time]
        hdm = hdm - np.nanmean(yaw[t-1:t:1])
        if len(hdm)==pre_time+post_time:
            head_STA.append(hdm)

    # peaks_neg = peaks_neg[peaks_neg+post_time < len(posL)]
    for t in peaks_neg:

        sac = posL[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:pre_time//2])
        if (np.isnan(sac).sum() < post_time) & (len(sac)==pre_time+post_time):
            eye_STA.append(sac)

        sac = -posR[t-pre_time : t+post_time]
        sac = sac - np.nanmean(sac[0:pre_time//2])
        if (np.isnan(sac).sum() < post_time) & (len(sac)==pre_time+post_time):
            eye_STA.append(sac)

        hdm = yaw[t-pre_time : t+post_time]
        hdm = hdm - np.nanmean(yaw[t-1:t:1])
        if len(hdm)==pre_time+post_time:
            head_STA.append(-hdm)

    return time_axis, eye_STA, head_STA