import numpy as np
import matplotlib.pyplot as plt

from utils.session_data import SessionData
from utils.saccade_triggered_average import _plot_mean_se, LE_COLOR, HEAD_COLOR

SAME_COLOR = "#4DAF4A"
OPP_COLOR  = "#984EA3"


def eye_head_correlogram(session, eye="LE", max_lag=120, bin_size=6):
    """Session-wide cross-correlogram of eye-saccade onsets relative to head-saccade onsets.

    For every head onset, collects all eye-saccade onsets within ±max_lag frames
    and bins the lag (eye_onset - head_onset). Also splits into same-direction
    and opposite-direction events using eye velocity sign vs head yaw sign.

    Returns:
        bin_centers: array of lag values in frames
        counts_all, counts_same, counts_opp: event counts per bin
    """
    df_eye  = session.df_LE if eye == "LE" else session.df_RE
    vx_arr  = np.array(session.LE_vx if eye == "LE" else session.RE_vx, dtype=float)
    yaw_v   = np.array(session.yaw_v, dtype=float)
    df_head = session.df_head

    eye_onsets  = df_eye["onset"].to_numpy()
    head_onsets = df_head["onset"].to_numpy()

    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    counts_all  = np.zeros(len(bins) - 1, dtype=int)
    counts_same = np.zeros(len(bins) - 1, dtype=int)
    counts_opp  = np.zeros(len(bins) - 1, dtype=int)

    for h_onset in head_onsets:
        # head direction: sign of mean yaw_v around onset
        v_window = slice(max(0, h_onset - 3), min(len(yaw_v), h_onset + 6))
        head_sign = np.sign(np.nanmean(yaw_v[v_window]))

        for e_onset in eye_onsets:
            dt = e_onset - h_onset
            if abs(dt) > max_lag:
                continue

            # eye direction: sign of eye velocity at onset
            eye_sign = np.sign(vx_arr[e_onset]) if np.isfinite(vx_arr[e_onset]) else 0.0

            idx = np.searchsorted(bins, dt, side="right") - 1
            if 0 <= idx < len(counts_all):
                counts_all[idx] += 1
                if eye_sign == head_sign:
                    counts_same[idx] += 1
                elif eye_sign != 0:
                    counts_opp[idx] += 1

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, counts_all, counts_same, counts_opp


def eye_head_coincidence(session, eye="LE", window=12, n_shuffles=1000):
    """Coincidence fractions with a circular-shuffle chance baseline.

    Returns a dict with:
        frac_head_with_eye: fraction of head saccades that have >=1 eye saccade within ±window
        frac_eye_with_head: fraction of eye saccades within ±window of any head onset
        chance_head_with_eye: shuffle mean for frac_head_with_eye
        chance_eye_with_head: shuffle mean for frac_eye_with_head
    """
    df_eye  = session.df_LE if eye == "LE" else session.df_RE
    df_head = session.df_head
    n_frames = len(session.LE_x)

    eye_onsets  = df_eye["onset"].to_numpy()
    head_onsets = df_head["onset"].to_numpy()

    def _fracs(e_on, h_on):
        fh = np.mean([np.any(np.abs(e_on - ho) <= window) for ho in h_on]) if len(h_on) else 0.0
        fe = np.mean([np.any(np.abs(h_on - eo) <= window) for eo in e_on]) if len(e_on) else 0.0
        return fh, fe

    obs_fh, obs_fe = _fracs(eye_onsets, head_onsets)

    rng = np.random.default_rng(0)
    ch_fh, ch_fe = [], []
    for _ in range(n_shuffles):
        shift = rng.integers(1, n_frames)
        shuffled = (eye_onsets + shift) % n_frames
        fh, fe = _fracs(shuffled, head_onsets)
        ch_fh.append(fh)
        ch_fe.append(fe)

    return {
        "frac_head_with_eye":   float(obs_fh),
        "frac_eye_with_head":   float(obs_fe),
        "chance_head_with_eye": float(np.mean(ch_fh)),
        "chance_eye_with_head": float(np.mean(ch_fe)),
    }


def plot_eye_head_correlogram(session, eye="LE", max_lag=120, bin_size=6, window=12, n_shuffles=1000):
    """Plot cross-correlogram of eye-saccade onsets relative to head-saccade onsets.

    Bars split by same-direction vs opposite-direction. Coincidence fractions
    with shuffle baseline annotated on the plot.

    Returns a matplotlib figure.
    """
    bin_centers, counts_all, counts_same, counts_opp = eye_head_correlogram(
        session, eye=eye, max_lag=max_lag, bin_size=bin_size)
    coinc = eye_head_coincidence(session, eye=eye, window=window, n_shuffles=n_shuffles)

    t_ms = bin_centers / 120 * 1000  # frames → ms

    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(t_ms, counts_all,  width=bin_size / 120 * 1000 * 0.9,
           color="gray", alpha=0.4, label="all")
    ax.bar(t_ms, counts_same, width=bin_size / 120 * 1000 * 0.9,
           color=SAME_COLOR, alpha=0.7, label="same dir")
    ax.bar(t_ms, counts_opp,  width=bin_size / 120 * 1000 * 0.9,
           color=OPP_COLOR, alpha=0.7, label="opp dir")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Eye onset − head onset (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"EO{session.eo}  {eye}", fontsize=7)
    ax.legend(fontsize=5, frameon=False)

    txt = (f"head+eye: {coinc['frac_head_with_eye']:.2f}  "
           f"eye+head: {coinc['frac_eye_with_head']:.2f} "
           f"(chance {coinc['chance_eye_with_head']:.2f})")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=5, va="top", ha="left")

    plt.tight_layout()
    return fig
