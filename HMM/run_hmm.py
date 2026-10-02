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
from hmm.features import session_features, fit_scaler, apply_scaler, mirror, FEATURE_NAMES
from hmm.splits import leave_one_session_out, leave_one_animal_out, select

cfg = load_config()
ids = eligible_sessions(cfg)
print(len(ids), "eligible sessions")


# %% load
q = cfg["qc"]
S = [load_session(s, max_speed=q["max_speed_mm_s"], max_abs_pitch=q["max_abs_pitch_deg"],
                  max_ang_speed=q["max_ang_speed_deg_s"], pad=q["pad_frames"]) for s in ids]


# %% QC: one row per session + sanity asserts
rows = []
for s in S:
    assert np.all(np.diff(s.t) > 0), s.session_id
    assert np.all((s.yaw_deg >= 0) & (s.yaw_deg < 360)), s.session_id
    assert len(s.yaw_deg) == len(s.t) == len(s.pos_xy) == len(s.valid), s.session_id

    dyaw = (np.diff(s.yaw_deg) + 180) % 360 - 180   # wrapped per-frame change
    dyaw = dyaw[s.valid[1:] & s.valid[:-1]]           # valid-to-valid frames only
    rows.append(dict(
        session_id=s.session_id, animal_id=s.animal_id, eo=s.eo,
        fs=s.fs, duration_s=s.t[-1] - s.t[0], frac_valid=s.valid.mean(),
        max_abs_dyaw_valid=np.abs(dyaw).max(),
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


# %% FEATURES: valid runs -> omega_yaw, log_speed, pitch at a common rate
f = cfg["features"]
segs = [g for s in S for g in session_features(s, **f)]

seg_df = pd.DataFrame([dict(session_id=g["session_id"], animal_id=g["animal_id"], eo=g["eo"], n=len(g["t"])) for g in segs])
summary = seg_df.groupby(["animal_id", "eo"]).agg(n_segs=("n", "size"), minutes=("n", "sum"))
summary["minutes"] = (summary["minutes"] / f["fs_out"] / 60).round(1)
print(summary)
print("total minutes:", round(seg_df.n.sum() / f["fs_out"] / 60, 1))


# %% FEATURES: inspect one segment
g = max(segs, key=lambda g: len(g["t"]))   # longest segment
fig, axes = plt.subplots(len(FEATURE_NAMES), 1, figsize=(10, 5), sharex=True)
for j, name in enumerate(FEATURE_NAMES):
    axes[j].plot(g["t"], g["X"][:, j], lw=0.5)
    axes[j].set_ylabel(name)
axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"{g['animal_id']}  EO{g['eo']}")
plt.show()


# %% FEATURES: distributions per animal (raw units: rad/s, log(mm/s+1), rad)
fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(10, 2.5))
for a in sorted({g["animal_id"] for g in segs}):
    Xa = np.concatenate([g["X"] for g in segs if g["animal_id"] == a])
    for j, name in enumerate(FEATURE_NAMES):
        axes[j].hist(Xa[:, j], bins=100, histtype="step", density=True, label=str(a))
        axes[j].set_title(name)
axes[0].legend()
plt.show()


# %% SPLITS + SCALING: leakage guard on one fold, mirror check
folds_loso = leave_one_session_out(segs)
folds_loao = leave_one_animal_out(segs)
print(len(folds_loso), "LOSO folds,", len(folds_loao), "LOAO folds")

train_ids, test_ids = folds_loso[0]
assert not set(train_ids) & set(test_ids)
train = select(segs, train_ids)
scaler = fit_scaler([g["X"] for g in train])                       # training sessions only
X_train = apply_scaler([g["X"] for g in train], scaler)
X_test = apply_scaler([g["X"] for g in select(segs, test_ids)], scaler)

# scaling stats unchanged if the held-out session is dropped from the pool -> no leakage
assert np.allclose(fit_scaler([g["X"] for g in segs if g["session_id"] in train_ids])["median"], scaler["median"])

# mirror: omega flips, others unchanged; augmented training set = original + mirrored
Xm = mirror(X_train[0])
assert np.allclose(Xm[:, 0], -X_train[0][:, 0]) and np.allclose(Xm[:, 1:], X_train[0][:, 1:])
X_train_aug = X_train + [mirror(X) for X in X_train]

# per-session drift of scaled features (median per session)
drift = pd.DataFrame([dict(session=f"{g['animal_id']} EO{g['eo']}", **dict(zip(FEATURE_NAMES, np.median(X, axis=0).round(2))))
                      for g, X in zip(train, X_train)]).groupby("session").median()
print(drift)
