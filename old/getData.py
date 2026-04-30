
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy import interpolate

# REMEMBER to upsample head/body data to match eye data sampling rate!
headSamplingRate = 90  # Hz
eyeSamplingRate = 120  # Hz
factor = eyeSamplingRate / headSamplingRate

# CAMERA PIXELS TO METER CONVERSION
behavCamScale = 11.5  # pixels per cm (From Emily 11-)

def getEyePos(df,eye='eye0',verbose=True):

    # get keypoint positions for specified eye from dataframe
    # filter x,y positions with a low-pass butterworth (use IIR low pass instead?)
    # average and return mean x y positions

    b, a = butter(N=3, Wn=0.3, btype='low') #cutoff: Wn * 120/2

    # find a video name that corresponds to the requested eye (e.g. "eye0DLC" for "eye0")
    videoNames = df['video'].unique()
    if eye not in videoNames:
        matches = [v for v in videoNames if eye.lower() in str(v).lower()]
        if matches:
            if verbose:
                print(f"Mapped requested eye '{eye}' -> video '{matches[0]}'")
            eye = matches[0]
        else:
            if verbose:
                print(f"No video matching '{eye}' found; continuing with eye='{eye}'")

    df_eye = df[
        (df['video'] == eye) &
          (df['processing_level'] == 'raw') #using raw becasue cleaned too filtered
                ].copy()

    keypoints = [f'p{i+1}' for i in range(8)]
    rawEyePos = {}
    for kp in keypoints:
        kp_data = df_eye[df_eye['keypoint'] == kp]
        x = kp_data['x'].to_numpy()
        y = kp_data['y'].to_numpy()
        rawEyePos[kp] = (x, y)

    x_stack = np.stack([xy[0] for xy in rawEyePos.values()])  # shape: (8, N)
    y_stack = np.stack([xy[1] for xy in rawEyePos.values()])  # shape: (8, N)

    # x_filtered = np.stack([medfilt(x, kernel_size=3) for x_kp in x_stack])
    x_filtered2 = np.stack([filtfilt(b, a, x_kp) for x_kp in x_stack])
    # y_filtered = np.stack([medfilt(y, kernel_size=3) for y_kp in y_stack])
    y_filtered2 = np.stack([filtfilt(b, a, y_kp) for y_kp in y_stack])

    xPos = np.mean(x_filtered2, axis=0)
    yPos = np.mean(y_filtered2, axis=0)

    ###############
    # convert eye position traces in pixels to estimated degrees of visual angle (dva)
    camera_width       = 1.15 # note: 1.15 mm sensor width at 200 Hz sampling (192 x 192 px)
    camera_sensor_size = 400 #images are 400 x 400 pixels at 120 Hz
    camera_lens_f      = 1.7 # mm
    camera_distance    = 24.1 # mm estimate via Jon working with 3D models -> see discord discussion
    eyeRadius          = 3.5 # mm

    #calculate image de-magnitifcation with thin lens equation
    #1/f = 1/d1 + 1/d2
    d2 = 1 / ( (1/camera_lens_f) - (1/camera_distance) )
    mag = d2/camera_distance
    #get resolution at eye with mag
    mm_per_pixel_at_eye = (camera_width/camera_sensor_size) / mag
    #estimate how many pixes in the image corresponds to a linear movement 
    # of the surface of the eye (arc length)
    deg_per_pixel = mm_per_pixel_at_eye / eyeRadius
    # convert from radians to degrees
    deg_per_pixel = (180/np.pi) * deg_per_pixel
    #apply conversion!
    xPos = xPos * deg_per_pixel
    yPos = yPos * deg_per_pixel
    ###############

    if verbose is True:
        t = np.arange(1,120*10)  # 10 seconds at 120 Hz
        plt.plot(t, xPos[t])
        plt.plot(t, yPos[t])

    timeStamps = df_eye.loc[df_eye['video'] == eye, 'timestamp'].unique()

    return xPos, yPos, timeStamps


def getHeadPos(df, eyeStartEnd, verbose=True):

    df_opt = df[df['data_type'] == 'optimized'].copy()
    # pivot so you have columns like 'x_nose', 'x_base'.
    df_wide = df_opt.pivot(index='frame', columns='marker', values=['x', 'y', 'z'])
    # flatten MultiIndex columns
    df_wide.columns = [f"{coord}_{marker}" for coord, marker in df_wide.columns]

    #compute average position of key markers
    x1 = df_wide['x_nose'].to_numpy()
    x2 = df_wide['x_base'].to_numpy()
    x3 = df_wide['x_left_cam_tip'].to_numpy()
    x4 = df_wide['x_right_cam_tip'].to_numpy()
    y1 = df_wide['y_nose'].to_numpy()
    y2 = df_wide['y_base'].to_numpy()
    y3 = df_wide['y_left_cam_tip'].to_numpy()
    y4 = df_wide['y_right_cam_tip'].to_numpy()
    # Compute average (or difference)
    xPos = (x1 + x2 + x3 + x4) / 4
    yPos = (y1 + y2 + y3 + y4) / 4

    #contrain to eyeStartEnd timestamps
    timeStamps = df_opt.timestamp.unique()
    mask = (timeStamps > eyeStartEnd[0]) & (timeStamps < eyeStartEnd[1])
    xPos = xPos[mask]
    yPos = yPos[mask]
    timeStamps = timeStamps[mask]

    # Upsample to match eye data rate (padded)
    xPos = upsample_wPad(xPos,factor)
    yPos = upsample_wPad(yPos,factor)

    #apply conversion from pixels to cm
    xPos = xPos / behavCamScale
    yPos = yPos / behavCamScale

    # linear velocity (world-frame)
    vx = np.gradient(xPos, 1/120)
    vy = np.gradient(yPos, 1/120)
    speed = np.sqrt(vx**2 + vy**2) 
    heading = np.arctan2(vy, vx)

    return xPos, yPos, speed, heading, timeStamps


def getHeadRot(df,eyeStartEnd):

    # extract DCM elements into a numpy array (N×3×3)
    R = df[[f"rotation_r{i}_c{j}" for i in range(3) for j in range(3)]].to_numpy()
    R = R.reshape(-1, 3, 3) # shape is now (N, 3, 3)

    #FILTER SHIT??!

    # Z–Y–X convention: yaw–pitch–roll
    pitch = np.arcsin(-R[:, 2, 0])                   # rotation about Y
    roll  = np.arctan2(R[:, 2, 1], R[:, 2, 2])       # rotation about X
    yaw   = np.arctan2(R[:, 1, 0], R[:, 0, 0])       # rotation about Z
    
    #contrain to eyeStartEnd timestamps
    timeStamps = df.timestamp.unique()
    mask = (timeStamps > eyeStartEnd[0]) & (timeStamps < eyeStartEnd[1])
    pitch = pitch[mask]
    roll = roll[mask]
    yaw = yaw[mask]
    timeStamps = timeStamps[mask]

    # Convert to degrees and upsample to match eye data rate
    pitch_deg = np.degrees( upsample_wPad(pitch,factor) )
    roll_deg  = np.degrees( upsample_wPad(roll,factor) )
    yaw_deg   = np.degrees( upsample_wPad(yaw,factor) )

    return yaw_deg, pitch_deg, roll_deg, timeStamps


def getToyPos(df,eyeStartEnd):
    # camera ID 24676894 is TOP DOWN view
    # TIDY CSV LABELS: 
    # video,frame,toy_top_x,toy_top_y,toy_tail_base_x,toy_tail_base_y,toy_nose_x,toy_nose_y,timestamps
    
            # df = df[df['video'] == '24676894_synchronized_corrected'].copy()
    df_opt = df[df['video'] == '24676894_synchronized_corrected'].copy()
    # pivot so you have columns like 'x_nose', 'x_base'.
    df_wide = df_opt.pivot(index='frame', columns='keypoint', values=['x', 'y'])
    # flatten MultiIndex columns
    df_wide.columns = [f"{coord}_{marker}" for coord, marker in df_wide.columns]
    #compute average position of key markers
    x1 = df_wide['x_toy_top'].to_numpy()
    x2 = df_wide['x_toy_tail_base'].to_numpy()
    x3 = df_wide['x_toy_nose'].to_numpy()
    y1 = df_wide['y_toy_top'].to_numpy()
    y2 = df_wide['y_toy_tail_base'].to_numpy()
    y3 = df_wide['y_toy_nose'].to_numpy()

    xPos = (x1 + x2 + x3) / 3
    yPos = (y1 + y2 + y3) / 3

    #contrain to eyeStartEnd timestamps
    timeStamps = df_opt[df_opt['keypoint']=='toy_nose'].timestamp.to_numpy()
    mask = (timeStamps > eyeStartEnd[0]) & (timeStamps < eyeStartEnd[1])
    xPos = xPos[mask]
    yPos = yPos[mask]
    timeStamps = timeStamps[mask]

    # Upsample to match eye data rate (padded)
    xPos = upsample_wPad(xPos,factor)
    yPos = upsample_wPad(yPos,factor)

    #apply conversion from pixels to meters
    xPos = xPos / behavCamScale
    yPos = yPos / behavCamScale

    return xPos, yPos, timeStamps


def getEyeConfidenced(df):
    # CSV lables: frames,camera,mean_confidence,timestamps,good_data,blink_threshold,confidence_threshold,eye_position_threshold
    # notes:
    # "good_data" is just the AND of the other boolean columns
    # if getAllFlags is True, return all boolean columns as separate dataframes
    # blink_threshold considers BOTH cameras already

    df_eye0 = df[df['camera'] == 'eye0'].copy()
    df_eye1 = df[df['camera'] == 'eye1'].copy()

    # if getAllFlags:
    #     bool_cols = ['good_data', 'blink_threshold', 'confidence_threshold', 'eye_position_threshold']
    #     df_eye0_bool = df_eye0[bool_cols].astype(bool)
    #     df_eye1_bool = df_eye1[bool_cols].astype(bool)
    # else:
    df_eye0_bool = df_eye0['good_data'].to_numpy().astype(bool)
    df_eye1_bool = df_eye1['good_data'].to_numpy().astype(bool)

    blinkThreshL = df_eye0['blink_threshold'].to_numpy().astype(bool)
    blinkThreshR = df_eye1['blink_threshold'].to_numpy().astype(bool)

    return df_eye0_bool, df_eye1_bool, blinkThreshL, blinkThreshR


########### util functions ########### 
def upsample_wPad(x,factor=120/90):

    #hard coded (should not need to change for now)
    pad_len=90
    kind="cubic"

    #pad signal to reduce edge artifacts
    x = np.asarray(x)
    x_pad = np.pad(x, (pad_len, pad_len), mode="reflect")
    
    n_old = len(x_pad)
    n_new = int(np.ceil(n_old * factor))
    old_idx = np.linspace(0, 1, n_old)
    new_idx = np.linspace(0, 1, n_new)
    
    #upsample
    interp_func = interpolate.interp1d(old_idx, x_pad, kind=kind)
    x_new = interp_func(new_idx)
    
    #remove pad
    pad_len_new = int(np.ceil(pad_len * factor))
    x_new = x_new[ pad_len_new : -pad_len_new ]

    return x_new


def alignEyeData(time_ref,time_other,value,verbose=True):

    mask = (time_ref >= time_other[0]) & (time_ref <= time_other[-1])
    aligned_v = np.full_like(time_ref, np.nan)
    aligned_v[mask] = np.interp(time_ref[mask], time_other, value)
     
    if verbose: 
            
        plt.figure(figsize=(10, 5))

        plt.plot(time_other, value, 'o-', label='original other eye', alpha=0.6)
        plt.plot(time_ref, aligned_v, '-', label='aligned/interpolated other eye', linewidth=2)

        plt.xlabel('Time (s)')
        plt.title('Alignment Check: Original vs. Interpolated')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return aligned_v