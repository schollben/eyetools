# %% init
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT + "/HMM")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmm.adapter import load_session
from hmm.sessions import load_config, eligible_sessions

cfg = load_config()
ids = eligible_sessions(cfg)
print(len(ids), "eligible sessions")


# %% load
S = [load_session(s, max_speed=cfg["qc"]["max_speed_mm_s"]) for s in ids]


# %% QC: one row per session + sanity asserts
rows = []
for s in S:
    assert np.all(np.diff(s.t) > 0), s.session_id
    assert np.all((s.yaw_deg >= 0) & (s.yaw_deg < 360)), s.session_id
    assert len(s.yaw_deg) == len(s.t) == len(s.pos_xy) == len(s.valid), s.session_id

    dyaw = (np.diff(s.yaw_deg) + 180) % 360 - 180   # wrapped per-frame change
    rows.append(dict(
        session_id=s.session_id, animal_id=s.animal_id, eo=s.eo,
        fs=s.fs, duration_s=s.t[-1] - s.t[0], frac_valid=s.valid.mean(),
        max_abs_dyaw=np.abs(dyaw).max(),
        x_range=np.ptp(s.pos_xy[:, 0]), y_range=np.ptp(s.pos_xy[:, 1]),
    ))

qc = pd.DataFrame(rows)
print(qc.drop(columns="session_id").round(2).to_string())
print("minutes per animal:\n", (qc.groupby("animal_id").duration_s.sum() / 60).round(1))


# %% inspect one session: yaw, speed, valid
n = 0
s = S[n]
speed = np.linalg.norm(s.vel_global, axis=1)

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(s.t, s.yaw_deg, lw=0.5)
axes[0].set_ylabel("yaw (deg)")
axes[1].plot(s.t, speed, lw=0.5)
axes[1].set_ylabel("speed (mm/s)")
axes[2].plot(s.t, s.valid, lw=0.5)
axes[2].set_ylabel("valid")
axes[2].set_xlabel("time (s)")
axes[0].set_title(f"{s.animal_id}  EO{s.eo}")
plt.show()
