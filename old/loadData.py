
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
import getData
from dataclasses import dataclass
#set up a dataclass to hold trial data
@dataclass
class TrialData:
    xL: np.ndarray
    yL: np.ndarray
    xR: np.ndarray
    yR: np.ndarray
    xH: np.ndarray
    yH: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    speed: np.array
    heading: np.array
    eyeConfL: np.ndarray
    eyeConfR: np.ndarray
    blinkThreshL: np.ndarray
    blinkThreshR: np.ndarray
    xToy: np.ndarray
    yToy: np.ndarray
    def __post_init__(self):
        #validation function as were pulling in data
        pass
# initialize results dictionary
results = {}

def loadData(paths:dict[str, str],
             applyNaN=True,
             verbose=True):

    for key in paths.keys():

        print('loading data for:', key)
        headrot = paths[key] + 'full_recording/rotation_translation_data.csv'
        head = paths[key] + 'full_recording/tidy_trajectory_data.csv'
        toy = paths[key] + 'full_recording/toy_2d_tidy.csv'
        eye = paths[key] + 'full_recording/eye_data.csv'
        eyeconf = paths[key] + 'full_recording/eye_model_v3_mean_confidence.csv'

        if os.path.exists(eye):
            df_eye = pd.read_csv(eye)
        else:
            df_eye = pd.DataFrame()
            print('no eye data??? check path:', eye)
            continue

        if os.path.exists(eyeconf):
            df_eyeconf = pd.read_csv(eyeconf)
        else:
            df_eyeconf = pd.DataFrame()

        if os.path.exists(headrot):
            df_headrot = pd.read_csv(headrot)
        else:
            df_headrot = pd.DataFrame()

        if os.path.exists(head):
            df_head = pd.read_csv(head)
        else:
            df_head = pd.DataFrame()

        if os.path.exists(toy):
            df_toy = pd.read_csv(toy)
        else:
            df_toy = pd.DataFrame()

        xL, yL, timeStampL = getData.getEyePos(df_eye, eye='eye0', verbose=False)
        xR, yR, timeStampR = getData.getEyePos(df_eye, eye='eye1', verbose=False)
        eyeConfL, eyeConfR, blinkThreshL, blinkThreshR = getData.getEyeConfidenced(df_eyeconf)

        # grab these values to contrain head/toy data extraction
        eyeStartEnd = [timeStampL[0],timeStampL[-1]]

        #head
        xH, yH, spd, hdn,timeStampsHead = getData.getHeadPos(df_head, eyeStartEnd)
        timeStampsHead = timeStampsHead - timeStampsHead[0]

        #head rotations (direction cosine matrix)
        yaw, pitch, roll, timeStampsHeadRot = getData.getHeadRot(df_headrot, eyeStartEnd)
        timeStampsHeadRot = timeStampsHeadRot - timeStampsHeadRot[0]

        #toy tracking data
        if not df_toy.empty:
            xToy, yToy, timeStampsToy = getData.getToyPos(df_toy, eyeStartEnd)
            timeStampsToy = timeStampsToy - timeStampsToy[0]
        else:
            xToy = np.array([])
            yToy = np.array([])
            timeStampsToy = np.array([])

        # align eye data based on timestamps (longer vector is reference)
        timeStampL = timeStampL - timeStampL[0]
        timeStampR = timeStampR - timeStampR[0]

        if len(timeStampL) != len(timeStampR):
            if len(timeStampL) > len(timeStampR):
                time_ref = timeStampL
                xR = getData.alignEyeData(time_ref,timeStampR,xR,verbose=False)
                yR = getData.alignEyeData(time_ref,timeStampR,yR,verbose=False)
                eyeConfR = getData.alignEyeData(time_ref,timeStampR,eyeConfR,verbose=False)
                blinkThreshR = getData.alignEyeData(time_ref,timeStampR,blinkThreshR,verbose=False)
            else: 
                time_ref = timeStampR
                xL = getData.alignEyeData(time_ref,timeStampL,xL,verbose=False)
                yL = getData.alignEyeData(time_ref,timeStampL,yL,verbose=False)
                eyeConfL = getData.alignEyeData(time_ref,timeStampL,eyeConfL,verbose=False)
                blinkThreshL = getData.alignEyeData(time_ref,timeStampL,blinkThreshL,verbose=False)

        # pad missing eye frames with NaNs (hopefully make same length as behavioral data)
        dt_eye = np.diff(time_ref)
        expected_dt = 1/120  # expected time difference between frames at 120 Hz
        gap_indices = np.where(dt_eye > 1.5 * expected_dt)[0] 

        if len(gap_indices) > 0:
            time_full = [time_ref[0]]
            for i in range(len(time_ref) - 1):
                time_full.append(time_ref[i + 1])
                if i in gap_indices:
                    # number of missing frames
                    n_missing = int(round(dt_eye[i] / expected_dt)) - 1
                    if n_missing > 0:
                        missing_ts = time_ref[i] + expected_dt * np.arange(1, n_missing + 1)
                        time_full[-1:-1] = missing_ts
            time_full = np.array(time_full, dtype=np.float64)

            # fill NaNs into eye data arrays where frames were dropped
            def pad_with_nans(original_time, full_time, data):
                data_full = np.full_like(full_time, np.nan, dtype=float)
                mask = np.isin(full_time, original_time)
                data_full[mask] = data
                return data_full

            # Update all signals that share this reference timeline
            xL = pad_with_nans(time_ref, time_full, xL)
            yL = pad_with_nans(time_ref, time_full, yL)
            xR = pad_with_nans(time_ref, time_full, xR)
            yR = pad_with_nans(time_ref, time_full, yR)
            blinkThreshR = pad_with_nans(time_ref, time_full, blinkThreshR)
            blinkThreshL = pad_with_nans(time_ref, time_full, blinkThreshL)
            time_ref = time_full  # replace old timeline with NaN-padded one

        if verbose:
            plt.figure()
            plt.plot(time_ref, label='eye time')
            plt.plot(timeStampsHead, label='head time')
            plt.grid()
            plt.legend()
            plt.show()

        print('data length (eyeL, eyeR, head, yaw, toy):', 
              len(xL), len(xR), len(xH), len(yaw), len(xToy),len(blinkThreshL),len(blinkThreshR))

        #resample head and toy data to match eye data length or vise versa
        # hopefully only need to do this as a minor fix for a few frames
        if len(xR) != len(xH):
            if len(xR) > len(xH):
                n_samples = len(xR)
                xH = resample(xH, n_samples)
                yH = resample(yH, n_samples)
                spd = resample(spd, n_samples)
                hdn = resample(hdn, n_samples)
                yaw = resample(yaw, n_samples)
                pitch = resample(pitch, n_samples)
                roll = resample(roll, n_samples)
                if not df_toy.empty: 
                    xToy = resample(xToy, n_samples)
                    yToy = resample(yToy, n_samples)
            else:
                n_samples = len(xH)
                xL = resample(xL, n_samples)
                yL = resample(yL, n_samples)
                xR = resample(xR, n_samples)
                yR = resample(yR, n_samples)
                eyeConfL = resample(eyeConfL, n_samples)
                eyeConfR = resample(eyeConfR, n_samples)
                blinkThreshL = resample(blinkThreshL, n_samples)
                blinkThreshR = resample(blinkThreshR, n_samples)

        print('after resampling:',
               len(xL), len(xR), len(xH), len(yaw), len(xToy),len(blinkThreshL),len(blinkThreshR))

        #set eye data values to NaN when blink threhold is 0
        if applyNaN:

            blinkThreshL = np.array(blinkThreshL, dtype=bool)
            blinkThreshR = np.array(blinkThreshR, dtype=bool)
            valid = np.logical_or(blinkThreshL, blinkThreshR) #valid = blinkThreshL & blinkThreshR
            len(valid)
            pad = 10 # expand each False region by 5 frames on each side
            expanded_invalid = np.convolve(~valid, np.ones(2*pad+1, dtype=int), mode='same') > 0
            valid = ~expanded_invalid

            xR = np.where(valid, xR, np.nan)
            yR = np.where(valid, yR, np.nan)
            xL = np.where(valid, xL, np.nan)
            yL = np.where(valid, yL, np.nan)


        #### OUTPUT DATA STRUCTURE ####
        results[key] = TrialData(xL, yL, xR, yR,
                                xH, yH, yaw, pitch, roll, spd, hdn, 
                                eyeConfL, eyeConfR, 
                                blinkThreshL, blinkThreshR,
                                xToy, yToy)
        


    return results