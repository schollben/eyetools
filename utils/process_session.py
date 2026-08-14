import numpy as np
from .extract_saccades import extract_saccades
from .ComputeWindowedRate import windowed_rate

def process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=2,
                    velocity_threshold_head=2, min_duration=12, min_inter_event=12):
    
    """Extract saccades, compute windowed rates, and attach NaN-masked
    speed / angVelocities / pupilSizes to R. Returns dfs + rate arrays."""
    
    df_LE = extract_saccades(R, 'eye', eye='LE', velocity_threshold=velocity_threshold_eye,
                             min_duration=min_duration, min_inter_event=min_inter_event)
    
    df_RE = extract_saccades(R, 'eye', eye='RE', velocity_threshold=velocity_threshold_eye,
                             min_duration=min_duration, min_inter_event=min_inter_event)
    
    df_head = extract_saccades(R, 'skull', velocity_threshold=velocity_threshold_head,
                               min_duration=min_duration, max_duration=600,
                               min_inter_event=min_inter_event)
    
    df_LEgaze = extract_saccades(R, 'gaze', eye='LE', velocity_threshold=velocity_threshold_gaze,
                                 min_duration=min_duration, min_inter_event=min_inter_event)
    
    df_REgaze = extract_saccades(R, 'gaze', eye='RE', velocity_threshold=velocity_threshold_gaze,
                                 min_duration=min_duration, min_inter_event=min_inter_event)

    n_frames = len(R.EQframes)

    # extraction params kept for downstream helpers (e.g. non_saccade_mask padding)
    R.min_inter_event = min_inter_event

    # saccade dataframes
    R.df_LE     = df_LE
    R.df_RE     = df_RE
    R.df_head   = df_head
    R.df_LEgaze = df_LEgaze
    R.df_REgaze = df_REgaze

    # windowed rates
    R.eyeRateLE  = windowed_rate(df_LE, n_frames, window_in_sec)
    R.eyeRateRE  = windowed_rate(df_RE, n_frames, window_in_sec)
    R.gazeRateLE = windowed_rate(df_LEgaze, n_frames, window_in_sec)
    R.gazeRateRE = windowed_rate(df_REgaze, n_frames, window_in_sec)

    # full-length kinematics with NaN where head-speed frames are masked out
    vv = np.sqrt(R.linearVel_x**2 + R.linearVel_y**2)
    indFrames = (vv >= 0) & (vv < 800)
    angVel = np.sqrt(R.roll_v**2 + R.pitch_v**2 + R.yaw_v**2)

    R.speed        = np.where(indFrames, vv, np.nan)
    R.angVelocities = np.where(indFrames, angVel, np.nan)