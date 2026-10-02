# HMM/ — how it works, how to run it, where it comes from

A reference for the `HMM/` subproject: unsupervised movement states from ferret head kinematics (sticky Gaussian HMM → sticky AR-HMM → HSMM).
The plan, decisions and results are in [plan.md](plan.md). This file covers running the code, understanding it and connecting it to the literature.

## Quick start
- **Environment:** conda env `eyetools-hmm` (a clone of `eyetools`, plus `ssm` and `ruptures`). Select it as the VS Code / Jupyter kernel.
- **Run:** open [run_hmm.py](run_hmm.py) and run its `# %%` cells top to bottom. Each cell has a 1–2 line comment saying what it does and where the method comes from.
- **Config:** everything that changes a result is in [config/hmm.yaml](config/hmm.yaml): cohort, QC thresholds, feature settings, grids, seeds.
- **Cache:** slow cells save to `HMM/results/` (gitignored). If the file exists they load it instead of refitting; delete the file to refit.
  - **Stale-cache gotcha:** cache file names contain K/L/κ but *not* the QC or feature settings. After changing `qc`, `features` or `sessions` in the config, delete `HMM/results/*` (or move it aside) before rerunning.
- **Progress:** every `Parallel(...)` call prints `Done X out of Y | elapsed … remaining …`. Cached cells print nothing and return instantly.
- **Headless:** to run end to end outside VS Code, drop the `%` magic lines, set `matplotlib.use("Agg")` and replace `plt.show()` with `savefig`.

## Data flow
```
utils.load_skull_data ──► adapter.load_session ──► Session (120 Hz, valid mask)
                                                     │
features.session_features ◄──────────────────────────┘
  valid runs ≥ 2 s → omega_yaw (SG deriv), log(speed+1), pitch → 12 Hz low-pass → 30 Hz
                                                     │
splits (LOSO / LOAO) → fit_scaler on train only → mirror augmentation (omega → −omega)
                                                     │
fit.fit_model (ssm: gaussian | ar | hsmm) → select.grid_scores (held-out LL / frame)
                                                     │
evaluate: reproducibility, generative checks, boundaries vs PELT, mirror pairs, describe
                                                     │
fit.save_run → results/<timestamp>_<run>/ (model, scaler, states + posteriors on original frames)
```

## File map
| File | What it does |
|---|---|
| [hmm/adapter.py](hmm/adapter.py) | `Session` dataclass; `load_session` wraps `utils.load_skull_data` and builds the `valid` mask (speed, \|pitch\|, \|ω\| glitches, padded) |
| [hmm/sessions.py](hmm/sessions.py) | `load_config`, `eligible_sessions` (cohort from config) |
| [hmm/features.py](hmm/features.py) | features, resampling, robust scaling (median/IQR), `mirror` |
| [hmm/splits.py](hmm/splits.py) | leave-one-session-out, leave-one-animal-out, `select` |
| [hmm/fit.py](hmm/fit.py) | `make_model`, `fit_model` (k-means init or warm start), `ll_per_frame`, `save_run` |
| [hmm/select.py](hmm/select.py) | `fold_data`, `grid_scores` (parallel CV grid), `best_restarts`, `plateau` (one-SE rule) |
| [hmm/evaluate.py](hmm/evaluate.py) | dwell times, Hungarian matching, mirror pairs, simulation, spectrum, PELT sweep, boundary agreement, mapping back to session frames, event table |
| [hmm/synth.py](hmm/synth.py) | synthetic data with known states, used to test the code |

## Learning path: our code → ssm source → paper
ssm is installed editable from `~/Documents/ssm`, so its source can be read and stepped through.

| Stage | Our code | Under the hood (ssm source) | Read |
|---|---|---|---|
| Features | [features.py](hmm/features.py) | scipy `savgol_filter`, `butter` / `sosfiltfilt`, `np.interp` | Savitzky & Golay 1964 |
| HMM basics (EM, Viterbi) | [fit.py](hmm/fit.py) | `ssm/hmm.py` (`fit`, `most_likely_states`, `expected_states`); `ssm/messages.py` (forward–backward, Viterbi) | Rabiner 1989; Murphy *PML* (HMM chapters) |
| Sticky κ | `make_model` | `StickyTransitions.m_step` in `ssm/transitions.py`: κ is added to the expected self-transition counts (about 10 lines) | Fox et al. 2011 |
| AR emissions | `make_model(kind="ar")` | `AutoRegressiveObservations` in `ssm/observations.py`: a weighted linear regression per state on L lagged frames | Wiltschko et al. 2015 |
| HSMM | `fit_model(kind="hsmm", init_from=…)` | `ssm/hsmm.py`, `NegativeBinomialSemiMarkovTransitions`: each state expands into r sub-states, so durations are negative-binomial | Johnson & Willsky 2013 |
| Model selection | [select.py](hmm/select.py) | — | Hastie et al. *ESL* §7.10 |
| Validation | [evaluate.py](hmm/evaluate.py) | `ruptures.Pelt`, `sklearn.metrics.adjusted_rand_score`, `scipy.optimize.linear_sum_assignment` | Killick et al. 2012; Hubert & Arabie 1985; Kuhn 1955 |

**Fastest way to build intuition: the SYNTH cells.** The true states are known there, so change one thing at a time and watch the result:
- κ in SYNTH 2 (0 vs 1e2 vs 1e6): decoded state durations;
- L=1 vs L=2: only the AR model with lags separates the 3 Hz oscillation from still;
- `dwell="gamma"` vs `"geometric"` in SYNTH 4: when the HSMM helps.

Two short pieces worth reading slowly:
- `StickyTransitions.m_step` in ssm, which is everything κ does;
- `mirror_pairs` in [evaluate.py](hmm/evaluate.py), which shows how left/right lives in the AR parameters (A → S A S, b → S b).

## Glossary
- **K:** number of discrete states.
- **L (lags):** how many past frames each AR state uses. L=3 at 30 Hz means 100 ms of history.
- **κ (kappa):** sticky prior, i.e. extra pseudo-counts on self-transitions. Larger κ means longer states in the *transition matrix*; decoded (Viterbi) durations can still be driven mostly by the emissions.
- **Held-out LL / frame (nats):** log-likelihood of held-out sessions divided by their frame count. Higher is better. Comparable across K/L/κ on the same data and features only.
- **Implied dwell:** mean duration the transition matrix predicts for state k, 1 / (1 − p_kk) frames. It is geometric by construction.
- **ARI (adjusted Rand index):** agreement between two labelings, invariant to label permutation. 1 = identical, 0 = chance.
- **Jaccard per state:** overlap of state k between two fits after Hungarian matching. Above 0.75 we call the state reproducible.
- **One-SE rule:** pick the simplest setting whose held-out score is within one (paired, across-fold) standard error of the best.
- **Mirror augmentation:** train on the data plus a copy with ω flipped, so left/right versions of each movement are learned symmetrically.
- **Circular-shift null:** rotate the state sequence within each segment. This keeps durations but breaks the alignment with the kinematics.

## Gotchas found so far
- **Building ssm on macOS:** a stale `/Library/Developer/CommandLineTools/usr/include/c++/v1` (2022) shadows the SDK's libc++ and gives `'cstdlib' file not found`.
  - Fix: move that directory aside, or build with `CPPFLAGS="-nostdinc++ -isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`.
  - ssm builds fine under numpy 2.
- **`import ssm` → "No module named ssm.cstats"** happens when you run Python from inside the ssm source directory. Run it from anywhere else.
- **`HSMM.fit` forwards fit kwargs into `initialize()`:** `fit.py` initializes first, then calls `fit(initialize=False)`.
- **HSMM from k-means lands in poor optima:** always warm-start it from a sticky AR-HMM (`init_from=`).
- **Euler yaw is already the nose heading:** the body x-axis is the nose, ZYX order. Yaw jumps come from steep pitch (gimbal lock) and tracking glitches; the `valid` mask handles both.
- **Very large κ (1e8):** the model then implies dwell times of hours, so simulations never switch. The GENERATIVE cell treats this as maximal misfit (KS = 1).

## References
- Rabiner LR (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc IEEE* 77:257–286.
- Murphy KP (2023). *Probabilistic Machine Learning: Advanced Topics.* MIT Press. (HMMs, SLDS, inference)
- Fox EB, Sudderth EB, Jordan MI, Willsky AS (2011). A sticky HDP-HMM with application to speaker diarization. *Ann Appl Stat* 5:1020–1056.
- Wiltschko AB et al. (2015). Mapping sub-second structure in mouse behavior. *Neuron* 88:1121–1135. (MoSeq, the AR-HMM for behavior; Datta lab)
- Weinreb C et al. (2024). Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nat Methods* 21:1329–1339.
- Johnson MJ, Willsky AS (2013). Bayesian nonparametric hidden semi-Markov models. *JMLR* 14:673–701.
- Linderman S et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems. *AISTATS.*
- ssm: github.com/lindermanlab/ssm (Linderman lab, Stanford). dynamax (JAX successor): github.com/probml/dynamax.
- Killick R, Fearnhead P, Eckley IA (2012). Optimal detection of changepoints with a linear computational cost. *JASA* 107:1590–1598. (PELT)
- Truong C, Oudre L, Vayatis N (2020). Selective review of offline change point detection methods. *Signal Processing* 167:107299. (ruptures)
- Hubert L, Arabie P (1985). Comparing partitions. *J Classification* 2:193–218. (ARI)
- Kuhn HW (1955). The Hungarian method for the assignment problem. *Naval Res Logist Q* 2:83–97.
- Savitzky A, Golay MJE (1964). Smoothing and differentiation of data by simplified least squares procedures. *Anal Chem* 36:1627–1639.
- Welch PD (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Trans Audio Electroacoust* 15:70–73.
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, 2nd ed., §7.10 (cross-validation, one-SE rule).
- Datta SR, Anderson DJ, Branson K, Perona P, Leifer A (2019). Computational neuroethology: a call to action. *Neuron* 104:11–24. (overview of the field)
- Alternatives (see plan.md, "Alternatives"):
  - Berman GJ et al. (2014), *J R Soc Interface* 11:20140672 (MotionMapper);
  - Hsu AI, Yttri EA (2021), *Nat Commun* 12:5188 (B-SOiD);
  - Luxem K et al. (2022), *Commun Biol* 5:1267 (VAME);
  - Ashwood ZC et al. (2022), *Nat Neurosci* 25:201–212 (GLM-HMM);
  - Tillmann JF et al. (2024), *Nat Methods* 21:703–711 (A-SOiD).
