import numpy as np
import matplotlib.pyplot as plt

from utils.session_data import SessionData


LE_COLOR   = "green"
RE_COLOR   = "#2166ac"
HEAD_COLOR = "#BE2323"


def _traces_for_eye(df, x_arr, pre, window, baseline_slice):
    """Align x_arr to onsets in df, baseline-subtract, and flip to positive."""
    traces = []
    n_frames = len(x_arr)
    for _, row in df.iterrows():
        onset = int(row["onset"])
        start, end = onset - pre, onset + window
        if start < 0 or end > n_frames:
            continue
        if not np.all(np.isfinite(x_arr[start:end])):
            continue
        tr = np.array(x_arr[start:end], dtype=float)
        tr -= np.mean(tr[baseline_slice])
        if np.nanmean(tr[20:min(40, len(tr))]) < 0:
            tr = -tr
        traces.append(tr)
    return traces


def _plot_mean_se(ax, t, traces, color, label):
    if not traces:
        return
    arr = np.array(traces)
    m  = np.nanmean(arr, axis=0)
    se = np.nanstd(arr,  axis=0) / np.sqrt(arr.shape[0])
    ax.plot(t, m, color=color, linewidth=1, label=f"{label} (n={arr.shape[0]})")
    ax.fill_between(t, m - se, m + se, color=color, alpha=0.3)


def _draw_saccade_triggered_average(ax, session, pre, window, eye_ylim, binocular):
    t = np.arange(-pre, window) / 120
    baseline = slice(pre - 2, pre)
    le_traces = _traces_for_eye(session.df_LE, session.LE_x, pre, window, baseline)
    re_traces = _traces_for_eye(session.df_RE, session.RE_x, pre, window, baseline)
    if binocular == "combined":
        _plot_mean_se(ax, t, le_traces + re_traces, LE_COLOR, "LE+RE")
    else:
        _plot_mean_se(ax, t, le_traces, LE_COLOR, "LE")
        _plot_mean_se(ax, t, re_traces, RE_COLOR, "RE")
    ax.set_ylim(-1, eye_ylim)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Eye position (deg)")
    ax.set_title(f"EO{session.eo}", fontsize=7)
    ax.legend(fontsize=5)


def saccade_triggered_average(
    sessions,
    window: int = 60,
    pre: int = 2,
    eye_ylim: float = 10,
    binocular: str = "separate",  # "separate" = LE and RE overlaid; "combined" = pool into one trace
):
    if not isinstance(sessions, list):
        sessions = [sessions]
    n = len(sessions)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2), squeeze=False)
    for ax, session in zip(axes[0], sessions):
        _draw_saccade_triggered_average(ax, session, pre, window, eye_ylim, binocular)
    plt.tight_layout()
    return fig


def _draw_saccade_andHead_triggered_average(ax_eye, session, pre, window, ylim, binocular):
    n_frames = len(session.LE_x)
    yaw = np.unwrap(np.array(session.yaw, dtype=float), period=360)
    baseline = slice(pre - 5, pre)
    t = np.arange(-pre, window) / 120
    ax_head = ax_eye.twinx()

    head_traces = []
    for _, row in session.df_head.iterrows():
        onset = int(row["onset"])
        start, end = onset - pre, onset + window
        if start < 0 or end > n_frames:
            continue
        if not np.all(np.isfinite(yaw[start:end])):
            continue
        h = yaw[start:end].copy()
        h -= np.mean(h[baseline])
        if np.mean(h[30:min(60, len(h))]) < 0:
            h = -h
        head_traces.append(h)

    le_traces = _traces_for_eye(session.df_LE, session.LE_x, pre, window, baseline)
    re_traces = _traces_for_eye(session.df_RE, session.RE_x, pre, window, baseline)

    if binocular == "combined":
        _plot_mean_se(ax_eye, t, le_traces + re_traces, LE_COLOR, "LE+RE")
    else:
        _plot_mean_se(ax_eye, t, le_traces, LE_COLOR, "LE")
        _plot_mean_se(ax_eye, t, re_traces, RE_COLOR, "RE")

    _plot_mean_se(ax_head, t, head_traces, HEAD_COLOR, "Head yaw")

    ax_eye.set_ylim(-1, ylim)
    ax_eye.set_xticks([0, window / 120])
    ax_eye.set_xlabel("Time (sec)")
    ax_eye.set_ylabel("Eye position (deg)")
    ax_head.set_ylabel("Head yaw (deg)", color=HEAD_COLOR)
    ax_head.tick_params(axis='y', colors=HEAD_COLOR)
    ax_eye.set_title(f"EO{session.eo}", fontsize=7)
    ax_eye.legend(fontsize=5)


def saccade_andHead_triggered_average(
    sessions,
    window: int = 60,
    pre: int = 10,
    ylim: float = 10,
    binocular: str = "separate",  # "separate" = LE and RE overlaid; "combined" = pool into one trace
):
    if not isinstance(sessions, list):
        sessions = [sessions]
    n = len(sessions)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3), squeeze=False)
    for ax, session in zip(axes[0], sessions):
        _draw_saccade_andHead_triggered_average(ax, session, pre, window, ylim, binocular)
    plt.tight_layout()
    return fig
