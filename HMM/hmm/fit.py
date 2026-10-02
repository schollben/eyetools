import copy
import hashlib
import pickle
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import ssm
import yaml


def make_model(kind: str, K: int, D: int, L: int = 1, kappa: float = 100, r_max: int = 10):
    """kind: 'gaussian' (sticky HMM), 'ar' (sticky AR-HMM) or 'hsmm' (AR emissions, NB dwell times)."""
    if kind == "gaussian":
        return ssm.HMM(K, D, observations="gaussian",
                       transitions="sticky", transition_kwargs=dict(kappa=kappa))
    if kind == "ar":
        return ssm.HMM(K, D, observations="ar", observation_kwargs=dict(lags=L),
                       transitions="sticky", transition_kwargs=dict(kappa=kappa))
    if kind == "hsmm":
        return ssm.HSMM(K, D, observations="ar", observation_kwargs=dict(lags=L),
                        transition_kwargs=dict(r_max=r_max))
    raise ValueError(f"unknown model kind: {kind}")


def fit_model(kind: str, Xs: list[np.ndarray], K: int, L: int = 1, kappa: float = 100,
              seed: int = 0, num_iters: int = 100, r_max: int = 10, init_from=None):
    """EM fit from a k-means initialization, or warm-started from the emission parameters of a
    fitted model (init_from, e.g. a sticky AR-HMM for an HSMM). Returns (model, log-prob per iteration)."""
    np.random.seed(seed)   # ssm and the k-means init use numpy's global RNG
    m = make_model(kind, K, Xs[0].shape[1], L, kappa, r_max)
    if init_from is not None:
        m.observations.params = copy.deepcopy(init_from.observations.params)
    else:
        m.initialize(Xs, init_method="kmeans")
    # initialize separately: HSMM.fit forwards fit kwargs into initialize()
    with np.errstate(divide="ignore"):   # HSMM transition matrices contain zeros
        lps = m.fit(Xs, method="em", num_iters=num_iters, initialize=False, verbose=0)
    return m, np.array(lps)


def ll_per_frame(m, Xs: list[np.ndarray]) -> float:
    """Held-out log-likelihood per frame (nats)."""
    with np.errstate(divide="ignore"):
        return m.log_likelihood(Xs) / sum(len(X) for X in Xs)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def save_run(name: str, m, scaler: dict, cfg: dict, params: dict,
             states: dict, posteriors: dict) -> Path:
    """Save a fitted model and per-session outputs to results/<run_id>/; append to results/runs.csv.
    params: model settings (kind, K, L, kappa, seed, ...); states/posteriors keyed by session_id."""
    cfg_txt = yaml.safe_dump(cfg, sort_keys=True)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    d = RESULTS_DIR / run_id
    d.mkdir(parents=True)

    with open(d / "model.pkl", "wb") as f:
        pickle.dump(m, f)
    np.savez(d / "scaler.npz", **scaler)
    np.savez_compressed(d / "states.npz", **states)
    np.savez_compressed(d / "posteriors.npz", **posteriors)
    (d / "config.yaml").write_text(cfg_txt)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=RESULTS_DIR.parent).stdout.strip()
    row = dict(run_id=run_id, config_hash=hashlib.sha1(cfg_txt.encode()).hexdigest()[:8], git=git,
               ssm=version("ssm"), numpy=np.__version__, **params)
    runs = RESULTS_DIR / "runs.csv"
    pd.DataFrame([row]).to_csv(runs, mode="a", header=not runs.exists(), index=False)
    return d
