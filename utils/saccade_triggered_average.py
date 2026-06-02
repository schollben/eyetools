import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.session_data import SessionData


def saccade_triggered_average(
    session: SessionData,
    df_saccades: pd.DataFrame,
    window: int = 60,
    pre: int = 2,
    eye_ylim: float = 10,
):
    eye_traces = []
    n_frames = len(session.LE_x)

    for _, row in df_saccades.iterrows():
        onset = int(row["onset"])
        start = onset - pre
        end = onset + window
        if start < 0 or end > n_frames:
            continue
        if not np.all(np.isfinite(session.LE_x[start:end])):
            continue

        eye = np.array(session.LE_x[start:end], dtype=float)
        eye -= eye[pre]
        # if np.cos(np.radians(row["direction_deg"])) < 0:
        #     eye = -eye
        if np.nanmean(eye[20:40]) < 0:
            eye = -eye  
        eye_traces.append(eye)

    eye_traces = np.array(eye_traces)   # (n_saccades, pre + window)
    n = eye_traces.shape[0]
    t = np.arange(-pre, window)/120

    eye_m  = np.nanmean(eye_traces, axis=0)
    eye_se = np.nanstd(eye_traces,  axis=0) / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(t, eye_m, color="green", linewidth=1, label="LE horizontal")
    ax.fill_between(t, eye_m - eye_se, eye_m + eye_se, color="green", alpha=0.3)
    ax.set_ylim(-1, eye_ylim)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Eye position (deg)")
    plt.tight_layout()

    return fig


def saccade_andHead_triggered_average(
    session: SessionData,
    df_head_saccades: pd.DataFrame,
    window: int = 60,
    pre: int = 10,
    ylim: float = 10,
):
    eye_traces, head_traces = [], []
    n_frames = len(session.LE_x)

    # unwrap full yaw signal to remove degree wrap-around discontinuities
    yaw = np.unwrap(np.array(session.yaw, dtype=float), period=360)

    for _, row in df_head_saccades.iterrows():
        onset = int(row["onset"])
        start = onset - pre
        end = onset + window
        if start < 0 or end > n_frames:
            continue
        if not np.all(np.isfinite(session.LE_x[start:end])):
            continue
        if not np.all(np.isfinite(yaw[start:end])):
            continue

        eye  = np.array(session.LE_x[start:end], dtype=float)
        head = yaw[start:end].copy()

        eye  -= np.mean(eye[pre - 5:pre])
        head -= np.mean(head[pre - 5:pre])

        if np.mean(head[30:60]) < 0:
            head = -head

        if np.mean(eye[20:40]) < 0:
            eye = -eye  

        eye_traces.append(eye)
        head_traces.append(head)

    eye_traces  = np.array(eye_traces)   # (n_saccades, pre + window)
    head_traces = np.array(head_traces)

    n = eye_traces.shape[0]
    t = np.arange(-pre, window)/120

    eye_m   = np.nanmean(eye_traces,   axis=0)
    eye_se  = np.nanstd(eye_traces,    axis=0) / np.sqrt(n)
    head_m  = np.nanmean(head_traces,  axis=0)
    head_se = np.nanstd(head_traces,   axis=0) / np.sqrt(n)

    fig, ax_eye = plt.subplots(figsize=(2, 3))
    ax_head = ax_eye.twinx()

    ax_eye.plot(t, eye_m,  color="green", linewidth=1, alpha=0.7, label="LE horizontal")
    ax_eye.fill_between(t, eye_m - eye_se, eye_m + eye_se, color="green", alpha=0.3)

    ax_head.plot(t, head_m, color="#BE2323", linewidth=1, alpha=0.7, label="Head yaw")
    ax_head.fill_between(t, head_m - head_se, head_m + head_se, color="#BE2323", alpha=0.3)

    ax_eye.set_ylim(-1, ylim)
    ax_eye.set_xticks([0, window/120])
    ax_eye.set_xlabel("Time (sec)")
    ax_eye.set_ylabel("Eye position (deg)", color="green")
    ax_head.set_ylabel("Head yaw (deg)", color="#BE2323")
    ax_eye.tick_params(axis='y', colors="green")
    ax_head.tick_params(axis='y', colors="#BE2323")

    lines = [plt.Line2D([0], [0], color="green", label="LE horizontal"),
             plt.Line2D([0], [0], color="#BE2323", label="Head yaw")]
    plt.tight_layout()

    return fig
