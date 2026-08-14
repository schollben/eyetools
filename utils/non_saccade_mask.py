import numpy as np


def non_saccade_mask(R, eyes=("LE", "RE"), pad_pre=3, pad_post=None):
    """Boolean mask (True = non-saccade frame) for VOR / fixation analysis.

    Excludes every frame inside an eye-saccade window from the requested eyes,
    padded to remove post-saccadic overshoot/settling. pad_post defaults to the
    min_inter_event used at extraction (falls back to 12). AND with
    np.isfinite(signal) at the call site for the signal you correlate.
    """
    n_frames = len(R.LE_vx)
    if pad_post is None:
        pad_post = getattr(R, "min_inter_event", 12)

    sacc = np.zeros(n_frames, bool)
    for eye in eyes:
        df = R.df_LE if eye == "LE" else R.df_RE
        for onset, peak in zip(df["onset"].to_numpy(), df["peak"].to_numpy()):
            sacc[max(0, onset - pad_pre):min(n_frames, peak + pad_post + 1)] = True

    return ~sacc
