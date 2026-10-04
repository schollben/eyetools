# %% init
# EXPLORATORY: the AR-HMM from run_hmm.py on head + eye + pupil features, older animals (EO 8-20). No model selection:
# fit a few K at L=3, kappa=100 and look at the states. Settings in HMM/config/hmm_eye.yaml; outputs in HMM/results_eye/.
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT + "/HMM")
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from hmm.adapter import load_session
from hmm.sessions import load_config, eligible_sessions, CONFIG_PATH
from hmm.features import session_features, fit_scaler, apply_scaler, mirror, EYE_FEATURE_NAMES, EYE_SIGNED
from hmm.fit import fit_model, ll_per_frame, RESULTS_DIR
from hmm.evaluate import run_lengths, match_states, mirror_pairs, event_table

cfg = load_config(CONFIG_PATH.parent / "hmm_eye.yaml")
ids = eligible_sessions(cfg)
print(len(ids), "eligible sessions")
OUT = RESULTS_DIR.parent / "results_eye"
OUT.mkdir(exist_ok=True)
NAMES = EYE_FEATURE_NAMES


def save(name):
    """Save the current figure to results_eye/<name>.png, then show it."""
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.show()


def plot_states(t, X, z, title="", seconds=60, fs=30):
    """Features (rows) under the decoded state sequence (colour band), first `seconds` of a segment."""
    n = min(len(t), int(seconds * fs))
    fig, axes = plt.subplots(len(NAMES) + 1, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.4] + [1] * len(NAMES)))
    axes[0].imshow(z[None, :n], aspect="auto", cmap="tab20", vmin=0, vmax=19,
                   interpolation="nearest", extent=[t[0], t[n - 1], 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title(title)
    for j, name in enumerate(NAMES):
        axes[j + 1].plot(t[:n], X[:n, j], lw=0.6, color="k")
        axes[j + 1].set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("time (s)")


# %% load
# Head as in run_hmm.py; eye via utils.load_session_data + removeBadData (LEQ / REQ masks). Eyes averaged in the head frame.
# Pupil: NaN where that eye's |x| or |y| >= 5 deg, % change from the session median, eyes averaged, gaps <= 2 s interpolated,
# 1 s moving average. Frames without eye or pupil data are invalid.
q = cfg["qc"]
S = [load_session(s, max_speed=q["max_speed_mm_s"], max_abs_pitch=q["max_abs_pitch_deg"],
                  max_ang_speed=q["max_ang_speed_deg_s"], pad=q["pad_frames"], eye=cfg["eye"]) for s in ids]


# %% QC: valid data and pupil coverage per session
# frac_valid: head + eye + pupil all usable. pupil_trusted_*: eye within ±5 deg (and good quality). pupil_filled: interpolated.
rows = []
for s in S:
    trusted = np.isfinite(s.pupil_eyes)
    either = trusted.any(axis=1)
    rows.append(dict(animal_id=s.animal_id, eo=s.eo, minutes=(s.t[-1] - s.t[0]) / 60, frac_valid=s.valid.mean(),
                     eye_finite=np.isfinite(s.eye_vx).mean(), pupil_trusted_LE=trusted[:, 0].mean(),
                     pupil_trusted_RE=trusted[:, 1].mean(), pupil_trusted_either=either.mean(),
                     pupil_filled=(np.isfinite(s.pupil) & ~either).mean(), pupil_nan=np.isnan(s.pupil).mean()))
qc = pd.DataFrame(rows)
print(qc.round(2).to_string())


# %% inspect pupil processing on one session (120 s)
# Eye position with the ±5 deg band, the gated per-eye pupil (gaps where the eye is off centre) and the final smoothed pupil.
s = S[0]
i0 = len(s.t) // 2
sl = slice(i0, i0 + int(120 * s.fs))
fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
for ax, v, name in [(axes[0], s.eye_x, "eye_x (deg)"), (axes[1], s.eye_y, "eye_y (deg)")]:
    ax.plot(s.t[sl], v[sl], lw=0.5, color="k")
    ax.axhspan(-cfg["eye"]["max_eye_deg"], cfg["eye"]["max_eye_deg"], color="g", alpha=0.15)
    ax.set_ylabel(name)
axes[2].plot(s.t[sl], s.pupil_eyes[sl, 0], lw=0.5, label="LE")
axes[2].plot(s.t[sl], s.pupil_eyes[sl, 1], lw=0.5, label="RE")
axes[2].set_ylabel("pupil, trusted (%)")
axes[2].legend(fontsize=7)
axes[3].plot(s.t[sl], s.pupil[sl], lw=1, color="k")
axes[3].set_ylabel("pupil, final (%)")
axes[3].set_xlabel("time (s)")
axes[0].set_title(f"{s.animal_id}  EO{s.eo}")
save("01_pupil_processing")


# %% FEATURES: head (5) + eye_x, eye_y (deg), eye_vx, eye_vy (signed log deg/s), pupil (% change) at 30 Hz
f = cfg["features"]
segs = [g for s in S for g in session_features(s, **f)]
seg_df = pd.DataFrame([dict(animal_id=g["animal_id"], eo=g["eo"], n=len(g["t"])) for g in segs])
summary = seg_df.groupby(["animal_id", "eo"]).agg(n_segs=("n", "size"), minutes=("n", "sum"))
summary["minutes"] = (summary["minutes"] / f["fs_out"] / 60).round(1)
print(summary)
print("total minutes:", round(seg_df.n.sum() / f["fs_out"] / 60, 1))

fig, axes = plt.subplots(2, 5, figsize=(16, 5))
for a in sorted({g["animal_id"] for g in segs}):
    Xa = np.concatenate([g["X"] for g in segs if g["animal_id"] == a])
    for j, name in enumerate(NAMES):
        axes.flat[j].hist(Xa[:, j], bins=100, histtype="step", density=True, label=str(a))
        axes.flat[j].set_title(name)
axes.flat[0].legend()
plt.tight_layout()
save("02_feature_distributions")


# %% SCALE: robust z-score over all sessions; mirror augmentation; data halves
sc_all = fit_scaler([g["X"] for g in segs])
X_all = apply_scaler([g["X"] for g in segs], sc_all)
Xm = mirror(X_all[0], EYE_SIGNED)
unsigned = [j for j in range(len(NAMES)) if j not in EYE_SIGNED]
assert np.allclose(Xm[:, EYE_SIGNED], -X_all[0][:, EYE_SIGNED]) and np.allclose(Xm[:, unsigned], X_all[0][:, unsigned])
X_all_aug = X_all + [mirror(X, EYE_SIGNED) for X in X_all]

halves = [[], []]   # alternate sessions (sorted by EO) within each animal
for a in sorted({g["animal_id"] for g in segs}):
    sess = sorted({(g["eo"], g["session_id"]) for g in segs if g["animal_id"] == a})
    for i, (_, sid) in enumerate(sess):
        halves[i % 2].append(sid)


# %% FIT: AR-HMM for each K (restarts on all data + one fit per half)  [cached]
ex, sel = cfg["explore"], cfg["selection"]
L, kappa = ex["L"], float(ex["kappa"])
seeds = list(range(cfg["seed"], cfg["seed"] + ex["n_restarts"]))
Xh = [[X for g, X in zip(segs, X_all) if g["session_id"] in h] for h in halves]
Xh = [Xs + [mirror(X, EYE_SIGNED) for X in Xs] for Xs in Xh]
todo = [K for K in ex["Ks"] if not (OUT / f"fits_K{K}.pkl").exists()]
jobs = [(K, "all", sd) for K in todo for sd in seeds] + [(K, h, cfg["seed"]) for K in todo for h in (0, 1)]
fits = Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(
    delayed(fit_model)("ar", X_all_aug if h == "all" else Xh[h], K, L=L, kappa=kappa, seed=sd, num_iters=sel["num_iters"])
    for K, h, sd in jobs)
for K in todo:
    pickle.dump(dict(models=[m for (k, h, _), (m, _) in zip(jobs, fits) if k == K and h == "all"],
                     halves=[m for (k, h, _), (m, _) in zip(jobs, fits) if k == K and h != "all"]),
                open(OUT / f"fits_K{K}.pkl", "wb"))
fits = {K: pickle.load(open(OUT / f"fits_K{K}.pkl", "rb")) for K in ex["Ks"]}


# %% DESCRIBE: per K — reproducibility, state table, state-mean heatmap, onset averages, 60 s raster, event table
raw_all = np.concatenate([g["X"] for g in segs])
Xs_flat = np.concatenate(X_all)
zs_by_K, rep_rows = {}, []
for K in ex["Ks"]:
    models, m_halves = fits[K]["models"], fits[K]["halves"]
    train_lls = [ll_per_frame(m, X_all_aug) for m in models]
    best = models[int(np.argmax(train_lls))]
    zs_all = [best.most_likely_states(X) for X in X_all]
    zs_by_K[K] = zs_all
    z_flat = np.concatenate(zs_all)

    # reproducibility: restarts, per-state Jaccard, halves
    z_cat = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in models]
    ari = np.mean([adjusted_rand_score(z_cat[i], z_cat[j]) for i in range(len(models)) for j in range(i + 1, len(models))])
    jacc = np.zeros((K, len(models)))
    for r, z in enumerate(z_cat):
        z = match_states(z_flat, z, K)[z]
        jacc[:, r] = [np.sum((z_flat == k) & (z == k)) / max(np.sum((z_flat == k) | (z == k)), 1) for k in range(K)]
    z_h = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in m_halves]
    rep_rows.append(dict(K=K, ari_restarts=ari, ari_halves=adjusted_rand_score(*z_h)))

    # state table (raw units)
    pair, _ = mirror_pairs(best, EYE_SIGNED)
    dw = [[] for _ in range(K)]
    for z in zs_all:
        st, a, b = run_lengths(z)
        for k, d in zip(st, b - a):
            dw[k].append(d / f["fs_out"])
    desc = pd.DataFrame({name: [raw_all[z_flat == k, j].mean() for k in range(K)] for j, name in enumerate(NAMES)})
    desc["occupancy"] = np.bincount(z_flat, minlength=K) / len(z_flat)
    desc["median_dwell_s"] = [np.median(d) if d else np.nan for d in dw]
    desc["mirror_partner"] = pair
    desc["jaccard_restarts"] = np.delete(jacc, int(np.argmax(train_lls)), axis=1).mean(1)
    print(f"\n=== K={K}: ARI restarts {ari:.2f}, halves {rep_rows[-1]['ari_halves']:.2f}")
    print(desc.round(2).to_string())

    # heatmap of state means (scaled units)
    M = np.array([Xs_flat[z_flat == k].mean(axis=0) for k in range(K)])
    fig, ax = plt.subplots(figsize=(9, 0.5 * K + 1.5))
    ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for k in range(K):
        for j in range(len(NAMES)):
            ax.text(j, k, f"{M[k, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(K), [f"{k} ({desc.occupancy[k]:.0%}, pair {pair[k]})" for k in range(K)])
    ax.set_title(f"K={K}: state means (robust z)")
    save(f"K{K}/14_state_means")
    desc.to_csv(OUT / f"K{K}" / "states.csv")

    # onset averages ±1 s (raw units)
    win = int(1 * f["fs_out"])
    lags = np.arange(-win, win) / f["fs_out"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for k in range(K):
        snips = []
        for g, z in zip(segs, zs_all):
            st, starts, _ = run_lengths(z)
            snips += [g["X"][a - win:a + win] for a in starts[(st == k) & (starts >= win) & (starts + win <= len(z))]]
        if snips:
            for j in range(len(NAMES)):
                axes.flat[j].plot(lags, np.mean(snips, axis=0)[:, j], color=plt.cm.tab20(k), label=f"{k} (n={len(snips)})")
    for j, name in enumerate(NAMES):
        axes.flat[j].axvline(0, color="k", lw=0.5)
        axes.flat[j].set_title(name)
    for ax in axes[1]:
        ax.set_xlabel("time from state onset (s)")
    axes.flat[-1].legend(fontsize=6)
    fig.suptitle(f"K={K}, L={L}, kappa={kappa:g}")
    plt.tight_layout()
    save(f"K{K}/15_onset_averages")

    # 60 s of the longest segment
    i = int(np.argmax([len(g["t"]) for g in segs]))
    plot_states(segs[i]["t"], segs[i]["X"], zs_all[i], title=f"K={K}, L={L}, kappa={kappa:g}")
    save(f"K{K}/16_states_60s")

    event_table(segs, zs_all, f["fs_out"]).to_csv(OUT / f"K{K}" / "events.csv", index=False)

rep = pd.DataFrame(rep_rows)
rep.to_csv(OUT / "reproducibility.csv", index=False)
print(rep.round(3))


# %% COMPARE: head-only K=4 model (results/..._ar_K4_L3_kappa100) decoded on the same frames vs each head+eye model
# Rows: head-only states; columns: head+eye states (row-normalised). Shows which head states the eye / pupil features split.
run_dir = sorted(RESULTS_DIR.glob("*_ar_K4_L3_kappa100"))[-1]
m_head = pickle.load(open(run_dir / "model.pkl", "rb"))
sc_head = dict(np.load(run_dir / "scaler.npz"))
z_head = np.concatenate([m_head.most_likely_states(X) for X in apply_scaler([g["X"][:, :5] for g in segs], sc_head)])
print("head-only state means (raw):")
print(pd.DataFrame({n: [raw_all[z_head == k, j].mean() for k in range(m_head.K)] for j, n in enumerate(NAMES[:5])}).round(2))

fig, axes = plt.subplots(1, len(ex["Ks"]), figsize=(5 * len(ex["Ks"]), 3))
for ax, K in zip(axes, ex["Ks"]):
    C = np.zeros((m_head.K, K))
    np.add.at(C, (z_head, np.concatenate(zs_by_K[K])), 1)
    C /= C.sum(axis=1, keepdims=True)
    ax.imshow(C, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for a in range(m_head.K):
        for b in range(K):
            ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=7, color="w" if C[a, b] < 0.5 else "k")
    ax.set_xlabel(f"head+eye state (K={K})")
    ax.set_ylabel("head-only state")
    ax.set_title(f"ARI {adjusted_rand_score(z_head, np.concatenate(zs_by_K[K])):.2f}")
plt.tight_layout()
save("17_head_vs_eye_states")


# =============================================================================
# SLOW (arousal) MODEL — 1 s bins, sticky Gaussian HMM. States are clusters of second-scale averages
# (pupil, speed, turning, eye movement, pitch); the full covariance gives the pupil x speed dependence within each state.
# =============================================================================

# %% SLOW: 1 s bins per valid segment -> pupil, log_speed, |omega_yaw|, log eye speed, pitch
sl = cfg["slow"]
nb = int(sl["bin_s"] * f["fs_out"])
c = {n: j for j, n in enumerate(NAMES)}
inv = lambda v: np.sign(v) * (np.exp(np.abs(v)) - 1)   # undo the signed log
SLOW_NAMES = ["pupil", "log_speed", "abs_omega", "log_eye_speed", "pitch"]
bins = []
for i, g in enumerate(segs):
    n = len(g["t"]) // nb
    if n < 2:
        continue
    X = g["X"][:n * nb].reshape(n, nb, -1)
    eye_speed = np.hypot(inv(X[..., c["eye_vx"]]), inv(X[..., c["eye_vy"]])).mean(1)   # deg/s
    B = np.column_stack([X[..., c["pupil"]].mean(1), X[..., c["log_speed"]].mean(1), np.abs(X[..., c["omega_yaw"]]).mean(1),
                         np.log(eye_speed + 1), X[..., c["pitch"]].mean(1)])
    bins.append(dict(seg=i, session_id=g["session_id"], animal_id=g["animal_id"], eo=g["eo"],
                     t=g["t"][:n * nb:nb] + sl["bin_s"] / 2, X=B))
Bc = np.concatenate([b["X"] for b in bins])
print(f"{len(Bc)} bins in {len(bins)} segments (median {np.median([len(b['X']) for b in bins]):.0f} bins per segment)")
print("correlations across bins:")
print(pd.DataFrame(np.corrcoef(Bc.T), index=SLOW_NAMES, columns=SLOW_NAMES).round(2))

# model-free look: pupil vs speed
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist2d(Bc[:, 1], Bc[:, 0], bins=50, cmap="Greys")
axes[0].set_xlabel("log_speed (1 s mean)")
axes[0].set_ylabel("pupil (% change)")
for a in sorted({b["animal_id"] for b in bins}):
    Ba = np.concatenate([b["X"] for b in bins if b["animal_id"] == a])
    axes[1].hist(Ba[:, 0], bins=60, histtype="step", density=True, label=str(a))
axes[1].set_xlabel("pupil (% change)")
axes[1].legend()
plt.tight_layout()
save("slow/20_pupil_vs_speed")


# %% SLOW: fit K = 2..6 (restarts on all bins + one fit per half); held-out LL across halves
sc_slow = fit_scaler([b["X"] for b in bins])
Xb = apply_scaler([b["X"] for b in bins], sc_slow)
Xbh = [[X for b, X in zip(bins, Xb) if b["session_id"] in h] for h in halves]
seeds_s = list(range(cfg["seed"], cfg["seed"] + sl["n_restarts"]))
jobs = [(K, "all", sd) for K in sl["Ks"] for sd in seeds_s] + [(K, h, cfg["seed"]) for K in sl["Ks"] for h in (0, 1)]
out = Parallel(n_jobs=sel["n_jobs"], verbose=0)(
    delayed(fit_model)("gaussian", Xb if h == "all" else Xbh[h], K, kappa=sl["kappa"], seed=sd, num_iters=sel["num_iters"])
    for K, h, sd in jobs)
slow = {K: dict(models=[m for (k, h, _), (m, _) in zip(jobs, out) if k == K and h == "all"],
                halves=[m for (k, h, _), (m, _) in zip(jobs, out) if k == K and h != "all"]) for K in sl["Ks"]}
pickle.dump(slow, open(OUT / "slow_fits.pkl", "wb"))

rows = []
for K in sl["Ks"]:
    ms, mh = slow[K]["models"], slow[K]["halves"]
    zc = [np.concatenate([m.most_likely_states(X) for X in Xb]) for m in ms]
    zh = [np.concatenate([m.most_likely_states(X) for X in Xb]) for m in mh]
    rows.append(dict(K=K, heldout_ll=(ll_per_frame(mh[0], Xbh[1]) + ll_per_frame(mh[1], Xbh[0])) / 2,
                     ari_restarts=np.mean([adjusted_rand_score(zc[i], zc[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]),
                     ari_halves=adjusted_rand_score(*zh)))
slow_rep = pd.DataFrame(rows)
slow_rep.to_csv(OUT / "slow" / "reproducibility.csv", index=False)
print(slow_rep.round(3))
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(slow_rep.K, slow_rep.heldout_ll, "o-")
axes[0].set_ylabel("held-out LL / bin (halves)")
axes[1].plot(slow_rep.K, slow_rep.ari_restarts, "o-", label="restarts")
axes[1].plot(slow_rep.K, slow_rep.ari_halves, "o-", label="halves")
axes[1].set_ylabel("ARI")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("K")
plt.tight_layout()
save("slow/21_K")


# %% SLOW: describe each K — state means, within-state pupil x speed correlation, dwell, movement make-up, time course
z_fast = zs_by_K[max(ex["Ks"])]   # fast AR-HMM states (largest K) for the movement make-up of each slow state
K_fast = max(ex["Ks"])
for K in sl["Ks"]:
    ms = slow[K]["models"]
    best_s = ms[int(np.argmax([ll_per_frame(m, Xb) for m in ms]))]
    zb = [best_s.most_likely_states(X) for X in Xb]
    zb_flat = np.concatenate(zb)
    S_ = best_s.observations.Sigmas
    dw = [[] for _ in range(K)]
    for z in zb:
        st, a, e = run_lengths(z)
        for k, d in zip(st, e - a):
            dw[k].append(d * sl["bin_s"])
    desc_s = pd.DataFrame({n: [Bc[zb_flat == k, j].mean() for k in range(K)] for j, n in enumerate(SLOW_NAMES)})
    desc_s["occupancy"] = np.bincount(zb_flat, minlength=K) / len(zb_flat)
    desc_s["r_pupil_speed"] = S_[:, 0, 1] / np.sqrt(S_[:, 0, 0] * S_[:, 1, 1])   # within-state correlation (model covariance)
    desc_s["median_dwell_s"] = [np.median(d) for d in dw]
    desc_s["mean_dwell_s"] = [np.mean(d) for d in dw]
    desc_s.to_csv(OUT / "slow" / f"K{K}_states.csv")
    print(f"\n=== slow K={K}")
    print(desc_s.round(2).to_string())

    # movement make-up: fraction of each slow state's 30 Hz frames in each fast state
    C = np.zeros((K, K_fast))
    for b, z in zip(bins, zb):
        np.add.at(C, (np.repeat(z, nb), z_fast[b["seg"]][:len(z) * nb]), 1)
    C /= C.sum(axis=1, keepdims=True)

    Zm = np.array([np.concatenate(Xb)[zb_flat == k].mean(axis=0) for k in range(K)])
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.6 * K + 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    axes[0].imshow(Zm, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for k in range(K):
        for j in range(len(SLOW_NAMES)):
            axes[0].text(j, k, f"{Zm[k, j]:.1f}", ha="center", va="center", fontsize=7)
    axes[0].set_xticks(range(len(SLOW_NAMES)), SLOW_NAMES, rotation=45, ha="right")
    axes[0].set_yticks(range(K), [f"{k} ({desc_s.occupancy[k]:.0%}, {desc_s.median_dwell_s[k]:.0f} s)" for k in range(K)])
    axes[0].set_title("state means (robust z)")
    for k in range(K):
        axes[1].scatter(Bc[zb_flat == k, 1], Bc[zb_flat == k, 0], s=3, color=plt.cm.tab10(k), label=str(k))
    axes[1].set_xlabel("log_speed (1 s mean)")
    axes[1].set_ylabel("pupil (% change)")
    axes[1].legend(markerscale=4, fontsize=7)
    axes[1].set_title("bins by state")
    axes[2].imshow(C, vmin=0, vmax=C.max(), cmap="viridis", aspect="auto")
    for k in range(K):
        for j in range(K_fast):
            axes[2].text(j, k, f"{C[k, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    axes[2].set_xlabel(f"fast state (AR-HMM K={K_fast})")
    axes[2].set_ylabel("slow state")
    axes[2].set_title("movement make-up")
    fig.suptitle(f"slow model K={K} (1 s bins, Gaussian, kappa={sl['kappa']})")
    plt.tight_layout()
    save(f"slow/K{K}_states")

    # time course: the session with the most bins, every bin coloured by state
    sid = max({b["session_id"] for b in bins}, key=lambda s: sum(len(b["X"]) for b in bins if b["session_id"] == s))
    fig, axes = plt.subplots(len(SLOW_NAMES), 1, figsize=(14, 8), sharex=True)
    for b, z in zip(bins, zb):
        if b["session_id"] == sid:
            for j, n in enumerate(SLOW_NAMES):
                axes[j].plot(b["t"], b["X"][:, j], color="0.8", lw=0.8)
                axes[j].scatter(b["t"], b["X"][:, j], c=[plt.cm.tab10(k) for k in z], s=8)
    for j, n in enumerate(SLOW_NAMES):
        axes[j].set_ylabel(n, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"slow K={K}: {sid.split('ferret_')[1].replace('_analyzable_output', '')}")
    save(f"slow/K{K}_timecourse")
