import numpy as np

def windowed_rate(df, n_frames, window_in_sec):

    """Saccade rate in each sliding window 
    for example: (events / 5s at 120Hz, window=600)."""

    samplingFreq = 120  # Hz
    window_len = int(window_in_sec * samplingFreq)
    starts = np.arange(0, n_frames - window_len + 1, window_len // 2)

    values = np.array([ (1 / window_len) * np.sum((df.onset >= s) & (df.onset < s + window_len)) for s in starts ])

    return values