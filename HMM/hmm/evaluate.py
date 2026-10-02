import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.optimize import linear_sum_assignment
from scipy.signal import welch


def run_lengths(z: np.ndarray):
    """State, start and stop index of each run in a state sequence."""
    starts = np.concatenate([[0], np.flatnonzero(np.diff(z)) + 1])
    stops = np.concatenate([starts[1:], [len(z)]])
    return z[starts], starts, stops


def dwell_times(zs: list[np.ndarray], K: int) -> list[np.ndarray]:
    """Run lengths (frames) per state, pooled over segments. The first and last run of each
    segment are censored by the segment edges and dropped."""
    out = [[] for _ in range(K)]
    for z in zs:
        states, starts, stops = run_lengths(z)
        for k, d in zip(states[1:-1], (stops - starts)[1:-1]):
            out[k].append(d)
    return [np.array(d) for d in out]


def transition_freqs(zs: list[np.ndarray], K: int) -> np.ndarray:
    """Fraction of state switches going k -> j (self-transitions excluded)."""
    C = np.zeros((K, K))
    for z in zs:
        states, _, _ = run_lengths(z)
        np.add.at(C, (states[:-1], states[1:]), 1)
    return C / C.sum()


def match_states(z_ref: np.ndarray, z: np.ndarray, K: int) -> np.ndarray:
    """Permutation perm such that perm[z] best overlaps z_ref (Hungarian on the confusion matrix)."""
    C = np.zeros((K, K))
    np.add.at(C, (z_ref, z), 1)
    rows, cols = linear_sum_assignment(-C)
    perm = np.empty(K, dtype=int)
    perm[cols] = rows
    return perm


def interior_accuracy(z_true: np.ndarray, z_pred: np.ndarray, K: int, margin: int = 3) -> float:
    """Accuracy after Hungarian matching, ignoring frames within `margin` of a true boundary."""
    perm = match_states(z_true, z_pred, K)
    near = np.zeros(len(z_true), bool)
    for b in np.flatnonzero(np.diff(z_true)) + 1:
        near[max(b - margin, 0):b + margin] = True
    return np.mean(perm[z_pred][~near] == z_true[~near])


def mirror_pairs(m, signed: list[int]):
    """Pair each state with the state whose emission parameters are its left/right mirror image.
    Returns (pair, dist): pair[k] is k's mirror partner (k itself for a symmetric state)."""
    obs, K, D = m.observations, m.K, m.D
    s = np.ones(D)
    s[signed] = -1
    if hasattr(obs, "As"):   # AR: x_t = sum_l A_l x_{t-l} + b; mirrored A_l -> S A_l S, b -> S b
        sL = np.tile(s, obs.lags)
        P = np.column_stack([obs.As.reshape(K, -1), obs.bs])
        Pm = np.column_stack([(s[None, :, None] * obs.As * sL[None, None, :]).reshape(K, -1), obs.bs * s])
    else:                    # Gaussian: mu -> S mu, Sigma -> S Sigma S
        P = np.column_stack([obs.mus, obs.Sigmas.reshape(K, -1)])
        Pm = np.column_stack([obs.mus * s, (s[None, :, None] * obs.Sigmas * s[None, None, :]).reshape(K, -1)])
    dist = np.linalg.norm(P[:, None, :] - Pm[None, :, :], axis=-1)
    _, pair = linear_sum_assignment(dist)
    return pair, dist[np.arange(K), pair]


def simulate(m, Ts: list[int], seed: int = 0):
    """Sample one state/feature sequence per length in Ts. Returns (zs, xs)."""
    np.random.seed(seed)
    out = [m.sample(T) for T in Ts]
    return [z for z, _ in out], [x for _, x in out]


def spectrum(xs: list[np.ndarray], col: int = 0, fs: float = 30, nperseg: int = 64):
    """Mean Welch power spectrum of one feature column over segments at least nperseg long."""
    P = [welch(x[:, col], fs=fs, nperseg=nperseg)[1] for x in xs if len(x) >= nperseg]
    return np.fft.rfftfreq(nperseg, 1 / fs), np.mean(P, axis=0)


def changepoint_sweep(Xs: list[np.ndarray], pens, min_size: int = 3, jump: int = 3) -> dict:
    """PELT (l2 cost) changepoints of each segment for each penalty: {pen: [indices per segment]}."""
    out = {pen: [] for pen in pens}
    for X in Xs:
        algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit(X)
        for pen in pens:
            out[pen].append(np.array(algo.predict(pen=pen)[:-1]))   # drop the end index
    return out


def boundary_agreement(zs: list[np.ndarray], cps: list[np.ndarray], tol: int = 3,
                       n_null: int = 200, seed: int = 0):
    """Fraction of state boundaries within ±tol frames of a changepoint, plus the same fraction
    for state sequences circularly shifted by a random offset within each segment (null)."""
    rng = np.random.default_rng(seed)

    def frac(shifts):
        hit = n = 0
        for z, cp, sh in zip(zs, cps, shifts):
            b = np.flatnonzero(np.diff(np.roll(z, sh))) + 1
            n += len(b)
            if len(b) and len(cp):
                hit += np.sum(np.abs(b[:, None] - cp[None, :]).min(axis=1) <= tol)
        return hit / n

    null = np.array([frac([rng.integers(len(z)) for z in zs]) for _ in range(n_null)])
    return frac([0] * len(zs)), null


def to_session_frames(s, segs: list[dict], vals: list[np.ndarray], fs_out: float, fill=-1) -> np.ndarray:
    """Map per-segment values on the fs_out grid (states, or posteriors T x K) back onto a
    Session's original timestamps; frames outside every segment get `fill`."""
    out = None
    for g, v in zip(segs, vals):
        if out is None:
            out = np.full((len(s.t),) + v.shape[1:], fill, dtype=v.dtype)
        if g["session_id"] != s.session_id:
            continue
        idx = np.flatnonzero((s.t >= g["t"][0]) & (s.t <= g["t"][-1]))
        out[idx] = v[np.clip(np.round((s.t[idx] - g["t"][0]) * fs_out).astype(int), 0, len(v) - 1)]
    return out


def event_table(segs: list[dict], zs: list[np.ndarray], fs_out: float) -> pd.DataFrame:
    """One row per state occurrence (state, t_start, t_stop, session_id), for video spot checks."""
    rows = []
    for g, z in zip(segs, zs):
        states, starts, stops = run_lengths(z)
        for k, a, b in zip(states, starts, stops):
            rows.append(dict(state=int(k), t_start=g["t"][a], t_stop=g["t"][b - 1] + 1 / fs_out,
                             session_id=g["session_id"]))
    return pd.DataFrame(rows)
