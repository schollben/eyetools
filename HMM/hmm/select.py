import itertools
import pandas as pd
from joblib import Parallel, delayed

from .features import fit_scaler, apply_scaler, mirror
from .splits import select
from .fit import fit_model, ll_per_frame


def fold_data(segs: list[dict], train_ids: list[str], test_ids: list[str], augment: bool = True):
    """Scaled train/test feature segments for one fold; scaler fit on training sessions only.
    augment: add left/right mirrored copies of the training segments."""
    train = [g["X"] for g in select(segs, train_ids)]
    test = [g["X"] for g in select(segs, test_ids)]
    scaler = fit_scaler(train)
    Xtr, Xte = apply_scaler(train, scaler), apply_scaler(test, scaler)
    if augment:
        Xtr = Xtr + [mirror(X) for X in Xtr]
    return Xtr, Xte, scaler


def _score(segs, fold, kind, K, L, kappa, seed, num_iters, augment, r_max):
    Xtr, Xte, _ = fold_data(segs, *fold, augment=augment)
    m, _ = fit_model(kind, Xtr, K, L=L, kappa=kappa, seed=seed, num_iters=num_iters, r_max=r_max)
    return dict(train_ll=ll_per_frame(m, Xtr), test_ll=ll_per_frame(m, Xte))


def grid_scores(segs, folds, kind, Ks, Ls=(1,), kappas=(100,), seeds=(0,), num_iters=100,
                augment=True, r_max=10, n_jobs=8) -> pd.DataFrame:
    """Held-out log-likelihood per frame for every fold x K x L x kappa x seed (fits run in parallel)."""
    combos = list(itertools.product(range(len(folds)), Ks, Ls, kappas, seeds))
    scores = Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        delayed(_score)(segs, folds[f], kind, K, L, kappa, seed, num_iters, augment, r_max)
        for f, K, L, kappa, seed in combos)
    return pd.DataFrame([dict(kind=kind, fold=f, K=K, L=L, kappa=kappa, seed=seed, **sc)
                         for (f, K, L, kappa, seed), sc in zip(combos, scores)])


def best_restarts(df: pd.DataFrame, by=("K", "L", "kappa")) -> pd.DataFrame:
    """Keep the restart with the highest training LL for each fold x setting."""
    return df.loc[df.groupby(["fold", *by]).train_ll.idxmax()]


def plateau(best: pd.DataFrame, col: str = "K") -> int:
    """Smallest value of `col` whose held-out LL is within one paired SE (across folds) of the best."""
    wide = best.pivot(index="fold", columns=col, values="test_ll")
    diff = wide.sub(wide[wide.mean().idxmax()], axis=0)   # per-fold difference to the best setting
    ok = diff.mean() >= -diff.sem()
    return ok[ok].index.min()
