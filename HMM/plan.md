# PLAN_HMM: unsupervised movement states from ferret head kinematics

Purpose: learn discrete movement states (sticky HMM → AR-HMM → HSMM) from existing head-tracking data, without hand labels, and validate them by label-free criteria.

**Build on the existing data-loading infrastructure. Do not write new loaders.**

## 0. Integration with existing infrastructure (do this first)
1. **Survey before writing code.** Read the repo's existing loading code and write a short summary into this file, under "Data contract (filled in)":
   - entry points (functions/classes), and how a session is identified (animal, date, session ID);
   - fields available: timestamps, yaw (global, 0–360°), global and local position/velocity, pitch/roll, z;
   - units, sampling rate, how missing frames are represented;
   - how neural data and tracking are aligned in time.
2. **Thin adapter only: `HMM/hmm/adapter.py`.** Write `load_session(session_id) -> Session`. It calls the existing loaders and maps their output onto the contract below.
   - Don't copy or modify the existing loaders. If a change is needed, propose it and ask first.
   - Reuse existing derived quantities (velocities, speed, pitch) instead of recomputing them, unless they're missing or noisy.
3. **`Session` contract** (dataclass):
   - **Required:** `session_id`, `animal_id`, `t` (s, monotonic), `fs`, `yaw_deg` (global, 0–360, NaN where missing), `pos_xy` (T×2).
   - **Optional:** `pitch_deg`, `roll_deg`, `pos_z`, `vel_global` (T×2), `vel_local` (T×2, frame documented), `units`, `neural` (handle to the existing neural loader plus a time-alignment function).
   - The adapter is responsible for:
     - resampling to a uniform grid if timestamps are irregular (yaw interpolated on the circle);
     - returning a boolean `valid` mask marking interpolated or dropped stretches.
4. **Session index: `HMM/hmm/sessions.py`.** List the eligible sessions from the existing metadata or database, not hard-coded paths, and keep the inclusion criteria in config.
5. **Accept:** the QC cell in `HMM/run_hmm.py` runs on all eligible sessions (fs, duration, fraction valid, max |Δyaw| per frame, position range) and its sanity asserts pass.

Data contract (filled in):
- **Entry points.**
  - `utils.load_skull_data(dir) -> dict` wraps bs `load_kinematics` and reads `skull_kinematics.csv`.
  - `utils.load_session_data(session)` also loads eye, gaze and toy data; it's heavier and not needed here.
  - `utils.get_sessions(*ferret_ids)` and `utils.parse_session_name(name) -> {session, date, id, age, eo}`.
  - Session ID = the directory name under `DATA_DIR` (`local_config.EYETOOLS_DATA_DIR`), e.g. `session_2025-10-17_ferret_420_E08_analyzable_output`.
- **Fields (`load_skull_data`):**
  - `skull_timestamps` (s, starting at 0);
  - `position_x/y` (mm, world);
  - `linearVel_x/y` (mm/s, world);
  - `roll/pitch/yaw` in degrees (Euler angles from arctan2, so yaw is in (−180, 180]; the adapter maps it to 0–360);
  - `roll_v/pitch_v/yaw_v`: rad/s, head-frame angular velocity.
  - z position and velocity, global angular velocity and keypoints are in the CSV but not returned.
- **Sampling.** Uniform within a session (dt jitter < 1e-6 s). fs differs between sessions (112.5–120.0 Hz), so features (§3) resample to a common rate rather than use an integer decimation factor.
- **Missing frames.** Skull data has no NaNs, because it is filled upstream. There's no head-quality flag; `valid` uses the existing criterion speed < 800 mm/s (`utils/process_session.py`). Eye quality flags (`LEQ`/`REQ`) exist for the eyes only.
- **Neural data.** None yet, so `Session.neural = None` and §6.5 is deferred.
- **Yaw gimbal lock (found in QC).** Euler yaw jumps 60–180° in a single frame whenever |pitch| ≈ 80–90° (head pointing straight down or up). This affects 5 of 10 sessions; 407 EO11 is worst, with 544 frames over 20°. §3 should not differentiate Euler yaw. Use a horizontal heading instead, e.g. atan2 of the head's forward vector projected onto the floor, or the global angular velocity about z, and consider masking frames with |pitch| > ~75°.

## 1. Approach
**Fitting is unsupervised.** Labels never enter any fit.
- Hand-built labels, if used at all, only *describe* states afterwards.
- Synthetic data tests the *code*, not the science.
- Scientific validation comes from held-out likelihood, reproducibility, generative checks and neural predictivity (§6).

**Models, in build order:**
1. **Sticky Gaussian HMM.** Each state has a fixed feature distribution, N(μ_k, Σ_k), and κ biases transitions toward staying put. This is the baseline.
2. **Sticky AR-HMM.** Each state is a linear dynamical system: x_t = A_k x_{t−L:t−1} + b_k + ε. States are defined by dynamics (oscillation, sustained rotation, transients). **This is the primary model.**
3. **HSMM.** Negative-binomial dwell times per state. Build it only if §6.3 shows the sticky HMM misfits the dwell times.
4. **Optional, later:** a sticky HDP-AR-HMM (`jax-moseq`, Gibbs sampling), which lets the data pick K.

## 2. Library: start with `ssm` (Linderman lab)
**Why `ssm`:**
- One API covers all of 1–3: `ssm.HMM(K, D, observations="gaussian"|"ar", transitions="sticky")` and `ssm.HSMM(K, D, observations=...)`.
- It takes a list of sessions of different lengths.
- It runs on CPU and fits with EM.
- `log_likelihood(x)` gives held-out scores.

**Install:** from source, in a dedicated environment (Python 3.10/3.11): `pip install numpy cython && pip install -e .` If the Cython build fails under numpy 2, pin `numpy<2`.

**Verify first:**
- the κ argument name (expected: `transition_kwargs=dict(kappa=...)`);
- whether `ssm.HSMM` accepts `observations="ar"`. If not, use Gaussian emissions on lag-stacked features [x_t, x_{t−1}].

**Fallbacks:**
- `dynamax` (JAX) if `ssm` won't build or is too slow. Sessions must be padded to equal length, and it has no HSMM.
- `jax-moseq` for the nonparametric fit. Check its input format: whitened data plus a mask.

## 3. Features: `hmm/features.py`
- **Input:** a `Session` from the adapter. Kinematics come from existing fields where possible; otherwise from `head_turn_segmentation.kinematics_global` (`win_slow=0.1` gives a 100 ms angular velocity).
- **Default feature set:**
  - ω_yaw (100 ms Savitzky–Golay);
  - log(speed + 1);
  - pitch, if available.
  - Optional additions: ω_rel (head rotation beyond the path), head_offset as sin/cos, local lateral/forward velocity.
- **No windowed summaries** such as R. The AR model should capture oscillation itself.
- **Downsample** to about 30 Hz with anti-aliasing (`scipy.signal.decimate`, integer factor). Keep an index map back to the original timestamps.
- **Scaling.** Robust z-score (median/IQR) pooled over the training sessions only, with the statistics saved alongside the model. Also keep per-session statistics so drift can be checked.
- **Handedness.**
  - Default: signed ω with **mirror augmentation** of the training sessions (ω → −ω, head_offset → −head_offset).
  - Afterwards, pair mirror states by sign-flipped emission parameters, and report both merged and per-direction results.
  - Compare against fitting |ω|.
- **Invalid frames:** split sessions into contiguous valid runs of at least 2 s, rather than interpolating across long gaps.

## 4. Splits: `hmm/splits.py`
- **Leave-one-session-out** for model selection.
- **Leave-one-animal-out** for generalization.
- Fix splits and seeds in `config/hmm.yaml`. Fit scaling statistics within the training folds only.

## 5. Fitting and selection: `hmm/fit.py`, `hmm/select.py`
**Milestone M1, baseline:**
- Sticky Gaussian HMM on one animal, K from 2 to 12.
- Held-out log-likelihood per frame against K.
- Plots of state sequences over the kinematics.

**Milestone M2, AR-HMM:**
- Grid: K from 2 to 20, L in {1, 2, 3}, κ on a log scale.
- At least 5 restarts per configuration.
- Select K at the elbow/plateau of the held-out likelihood, then confirm with leave-one-animal-out.

**Timescale reference:**
- Run `ruptures` PELT on the features, sweeping the penalty, and take the changepoint duration distribution at the plateau.
- Use it to choose κ: the median state duration should roughly match the median changepoint duration.

**Milestone M3, HSMM:** only if it's justified by §6.3. Compare against the sticky AR-HMM on the same folds.

**Save to `results/hmm/<run_id>/`, keyed by the existing session IDs:**
- parameters and scaling statistics;
- Viterbi states and posterior marginals, mapped back to original timestamps;
- the config and its hash.

Append each run (config, seed, library versions, git hash) to `results/hmm/runs.csv`.

## 6. Validation without hand labels: `hmm/evaluate.py`
1. **Predictive.** Held-out log-likelihood, both held-out-session and held-out-animal.
2. **Reproducibility.**
   - Match states across restarts and folds (Hungarian algorithm on emission parameters).
   - Report the adjusted Rand index across restarts and across data halves.
   - Don't interpret states that don't reproduce.
3. **Generative checks.** Simulate sessions from the fitted model and compare with real data:
   - per-state dwell-time distributions;
   - the ω power spectrum;
   - transition frequencies.
   This decides whether the HSMM is needed.
4. **Boundary agreement.** The fraction of state boundaries within ±100 ms of a model-free changepoint, against a circular-shift null.
5. **Neural predictivity.** Requires the existing neural loader plus time alignment. Cross-validated encoding models (ridge, or Poisson on inferred spikes), using the same folds, with nested predictors:
   - (a) continuous kinematics with lags;
   - (b) (a) + state indicators;
   - (c) (a) + state × kinematics interactions.
   States matter if (b) or (c) improves held-out variance explained beyond (a). Report this per neuron and as the fraction of neurons improved, against a circular-shift null.
6. **Description.**
   - State-triggered averages of the kinematics.
   - An event table (state, t_start, t_stop, session_id) for video spot checks.

## 7. Checks: cells in `HMM/run_hmm.py` (no pytest suite; keep it simple)
- **Synthetic `Session` generator.** Multiple sessions with randomized movement-type order and durations; optionally gamma-distributed dwell times. Reuse `synth.py` if it's present.
- **Accept:**
  - Adapter round trip: a synthetic `Session` → features → fit → states mapped back to the original timestamps.
  - AR-HMM at K = 6 recovers the synthetic states: interior-frame accuracy ≥ 0.85 after Hungarian matching.
  - Mirror augmentation produces sign-paired states.
  - With gamma dwell times, the HSMM beats the sticky HMM on held-out likelihood.
  - Leakage guard: no held-out session contributes to the fit or to the scaling statistics.
- Checks are plain asserts and plots in cells; keep long fits in their own cells.

## Layout (self-contained in `HMM/`, conda env `eyetools-hmm`)
```
HMM/hmm/adapter.py      Session contract + wrapper over existing loaders
HMM/hmm/sessions.py     session index from existing metadata
HMM/hmm/features.py     feature extraction, downsampling, scaling, mirroring
HMM/hmm/splits.py       CV folds
HMM/hmm/fit.py          ssm fitting (Gaussian, AR, HSMM)
HMM/hmm/select.py       grids, held-out scoring, K/L/κ selection
HMM/hmm/evaluate.py     validation (§6)
HMM/config/hmm.yaml     sessions, features, grids, folds, seeds, paths
HMM/run_hmm.py          # %% cell script: QC, fit, run, examine each method
HMM/results/            outputs (gitignored)
```

## Minimal starting sketch (M1/M2)
```python
import ssm
from hmm.adapter import load_session
from hmm.features import make_features            # returns list of (T_i x D), scaler
X_train, scaler = make_features([load_session(s) for s in train_ids], fit_scaler=True)
X_test, _ = make_features([load_session(s) for s in test_ids], scaler=scaler)
m = ssm.HMM(8, X_train[0].shape[1], observations="ar", transitions="sticky")
m.fit(X_train, method="em", num_iters=100)
ll = sum(m.log_likelihood(x) for x in X_test) / sum(len(x) for x in X_test)
z = [m.most_likely_states(x) for x in X_test]
```

## Conventions
- Re-run the relevant `run_hmm.py` cells after each change. Never tune on test folds.
- Use seconds for time and radians internally; convert to degrees only for display.
- **Model selection uses held-out likelihood and reproducibility only.**
- Don't loosen test tolerances without saying so.
- The user is experienced in Python and MATLAB. Explain choices concisely and skip boilerplate.

## Open questions (answered in §0)
- **Loaders and session ID:** `utils/` (see Data contract); the session ID is the directory name.
- **Neural data:** none yet, so §6.5 isn't feasible now.
- **Eye tracking:** available (left and right eye kinematics plus quality masks), so eye velocity is a possible later feature.
- **Data volume:** the default cohort (ferrets 402/405/407/420, EO ≥ 8; 402 and 405 have no sessions above EO7) is 10 sessions from 2 animals, about 44 min: 420 has 31 min, 407 has 13 min. Prefer the Gaussian HMM with small K and L = 1. Leave-one-animal-out has only 2 folds.
- **Missing modules:** `head_turn_segmentation` and `synth.py` don't exist, so features compute ω_yaw directly and the synthetic generator is written fresh.