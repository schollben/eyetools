import numpy as np
from scipy.ndimage import binary_dilation, label

def removeBadData(Results):

    #left eye
    LEQ = Results.LEQ.astype(bool)
    LEQ = ~binary_dilation(~LEQ, iterations = 6) # pyright: ignore[reportOperatorIssue]
    badFrames = np.where(~LEQ)[0]
    idx = np.where((np.diff(badFrames) > 1) & (np.diff(badFrames) < 60))[0]
    for i in idx:
        LEQ[badFrames[i]:badFrames[i+1]] = False

    Results.LE_x[~LEQ] = np.nan
    Results.LE_y[~LEQ] = np.nan
    Results.LE_vx[~LEQ] = np.nan
    Results.LE_vy[~LEQ] = np.nan
    Results.LE_ax[~LEQ] = np.nan
    Results.LE_ay[~LEQ] = np.nan
    Results.LE_pupil[~LEQ] = np.nan

    Results.LE_gaze_horizontal_deg[~LEQ] = np.nan
    Results.LE_gaze_vertical_deg[~LEQ] = np.nan
    Results.LE_ang_vel_local_x_deg_s[~LEQ] = np.nan
    Results.LE_ang_vel_local_y_deg_s[~LEQ] = np.nan

    #right eye
    REQ = Results.REQ.astype(bool) 
    REQ = ~binary_dilation(~REQ, iterations = 6) # pyright: ignore[reportOperatorIssue]
    badFrames = np.where(~REQ)[0]
    idx = np.where((np.diff(badFrames) > 1) & (np.diff(badFrames) < 60))[0]
    for i in idx:
        REQ[badFrames[i]:badFrames[i+1]] = False

    Results.RE_x[~REQ] = np.nan
    Results.RE_y[~REQ] = np.nan
    Results.RE_vx[~REQ] = np.nan
    Results.RE_vy[~REQ] = np.nan
    Results.RE_ax[~REQ] = np.nan
    Results.RE_ay[~REQ] = np.nan
    Results.RE_pupil[~REQ] = np.nan

    Results.RE_gaze_horizontal_deg[~REQ] = np.nan
    Results.RE_gaze_vertical_deg[~REQ] = np.nan
    Results.RE_ang_vel_local_x_deg_s[~REQ] = np.nan
    Results.RE_ang_vel_local_y_deg_s[~REQ] = np.nan