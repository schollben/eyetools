import numpy as np

from .adapter import Session

# six movement types in (omega_yaw, log_speed, pitch), scaled units
MEANS = np.array([
    [0.0, -1.0, 0.0],    # 0 still
    [1.5, 0.0, 0.0],     # 1 turn left
    [-1.5, 0.0, 0.0],    # 2 turn right (mirror of 1)
    [0.0, -1.0, 0.0],    # 3 head oscillation: same mean as still, differs only in dynamics
    [0.0, 1.5, -0.5],    # 4 locomotion
    [0.0, -0.5, 1.5],    # 5 pitch up
])


def _dynamics(fs: float = 30, f_osc: float = 3, r: float = 0.95):
    """Per-state AR(2) matrices; state 3 oscillates in omega at f_osc Hz."""
    A1 = np.stack([0.8 * np.eye(3) for _ in MEANS])
    A2 = np.zeros_like(A1)
    th = 2 * np.pi * f_osc / fs
    A1[3, 0, 0], A2[3, 0, 0] = 2 * r * np.cos(th), -r ** 2
    return A1, A2


def synth_segments(n_segments: int = 10, T: int = 3000, dwell: str = "geometric",
                   mean_dwell: float = 30, gamma_shape: float = 5, noise: float = 0.2, seed: int = 0):
    """Feature segments from 6 movement types in random order with random dwell times.
    dwell: 'geometric' or 'gamma' (frames, mean = mean_dwell). Returns (zs, Xs)."""
    rng = np.random.default_rng(seed)
    A1, A2 = _dynamics()
    K = len(MEANS)
    zs, Xs = [], []
    for _ in range(n_segments):
        z = np.empty(T, dtype=int)
        t, k = 0, rng.integers(K)
        while t < T:
            if dwell == "gamma":
                d = max(1, int(round(rng.gamma(gamma_shape, mean_dwell / gamma_shape))))
            else:
                d = rng.geometric(1 / mean_dwell)
            z[t:t + d] = k
            t += d
            k = rng.choice([j for j in range(K) if j != k])

        X = np.zeros((T, 3))
        x1 = x2 = MEANS[z[0]]
        for t in range(T):
            mu = MEANS[z[t]]
            x = mu + A1[z[t]] @ (x1 - mu) + A2[z[t]] @ (x2 - mu) + noise * rng.standard_normal(3)
            X[t], x2, x1 = x, x1, x
        zs.append(z)
        Xs.append(X)
    return zs, Xs


def synth_session(duration_s: float = 300, fs: float = 120, mean_dwell_s: float = 2, seed: int = 0):
    """A synthetic Session at fs with 3 movement types (still, turn left, turn right).
    Returns (Session, true state per frame)."""
    rng = np.random.default_rng(seed)
    T = int(duration_s * fs)
    z = np.empty(T, dtype=int)
    t, k = 0, 0
    while t < T:
        d = rng.geometric(1 / (mean_dwell_s * fs))
        z[t:t + d] = k
        t += d
        k = rng.choice([j for j in range(3) if j != k])

    omega_dps = np.array([0, 120, -120])[z] + 10 * rng.standard_normal(T)      # deg/s
    speed = np.array([5, 50, 50])[z] * np.exp(0.2 * rng.standard_normal(T))   # mm/s
    yaw = np.cumsum(omega_dps) / fs
    vel = speed[:, None] * np.column_stack([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
    s = Session(session_id="synthetic", animal_id=0, eo=0, t=np.arange(T) / fs, fs=fs,
                yaw_deg=yaw % 360, pos_xy=np.cumsum(vel, axis=0) / fs, valid=np.ones(T, bool),
                pitch_deg=np.array([0, 20, 20])[z] + 2 * rng.standard_normal(T), vel_global=vel)
    return s, z
