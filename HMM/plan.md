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
- **Yaw jumps (found in QC).** The skull x-axis is exactly the nose direction, and the Euler order is ZYX. So Euler yaw already is the floor-projected nose heading, and recomputing heading from quaternions changes nothing. Jumps come from two sources:
  - steep pitch, where heading is ill-conditioned: `valid` excludes |pitch| ≥ 75°;
  - tracking glitches: `valid` excludes |ω| ≥ 1500°/s. In clean sessions |ω| peaks at about 1440°/s; glitches reach 3700–10,000°/s.
  - Excluded stretches are padded by 6 frames, as in `removeBadData`.
  - 407 EO11 is only 68% valid after this and is a candidate for exclusion.

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

## Results, first full run (2026-10-01; cohort 402/405/407/420, EO ≥ 8, minus 407 EO11: 9 sessions, 2 animals, 41 min)

**Library checks (§2).**
- `ssm` builds under numpy 2.3; there's no need for `numpy<2`.
- Stickiness argument: `transition_kwargs=dict(kappa=...)`.
- `ssm.HSMM` accepts `observations="ar"`.
- `HSMM.fit` forwards fit kwargs into `initialize()`, so `fit.py` initializes separately.
- The HSMM must be warm-started from the sticky AR-HMM's emissions. From its own k-means start it lands in worse optima and doesn't beat the sticky model.

**Synthetic checks (§7): all pass.**
- Round trip: accuracy 1.0 on the original 120 Hz frames.
- AR-HMM K=6 recovery: 0.99 (Gaussian HMM 0.82; it can't separate oscillation from still).
- Mirror augmentation gives a turn-left/turn-right pair.
- Gamma dwell times: the HSMM beats the sticky AR-HMM by +0.02 nats/frame. Geometric dwell times (control): 0.00.

**Timescale (§5).**
- PELT changepoint rate falls steadily with penalty; there's no plateau.
- The elbow is at penalty 100, with a median changepoint interval of 2.25 s.

**κ.**
- Matching the median Viterbi state duration to 2.25 s pushes κ to the top of the grid (1e8 gives only 1.03 s).
- At that κ the transition matrix implies dwell times of hours, while decoded states last about 1 s.
- The κ-by-Viterbi-duration rule therefore fails here: decoding is driven by the emissions whatever the prior. **Needs a decision.** The DECIDE K / DECIDE KAPPA cells in `run_hmm.py` produce the plots (`results/decide_K.png`, `results/decide_kappa.png`).

**M1 (Gaussian HMM, ferret 420).** Held-out LL rises steadily from K=2 to 12, with no plateau.

**M2 (AR-HMM, LOSO, κ=1e8).**
- The number of lags dominates: L=3 is about 2 nats/frame better than L=1.
- For L=3, gains shrink after K≈6: K 2→4 +0.55, 4→6 +0.12, 6→8 +0.04, 8→20 +0.27 in total.
- The paired one-SE rule picks the grid edge (K=20).
- Leave-one-animal-out: both held-out ferrets level off at K≈6.

**Validation of K=20, L=3.**
- Agreement: ARI 0.58 across restarts, 0.34 across data halves. Only 5 of 20 states have Jaccard > 0.75 across restarts, so most states don't reproduce.
- Boundaries: 10.6% fall within ±100 ms of a changepoint, against 4.9% for the circular-shift null (p < 0.005).
- Mirror pairing: 3 left/right pairs, giving 17 merged states. ARI with a |ω| model is 0.36.
- Generative check: the decoded dwell distributions peak at 0.5–1 s with CV < 1, which is not geometric, while the model implies hours. The sticky HMM misfits dwell times, so §6.3 says to build the HSMM (M3, not yet run).

**Outputs:** `HMM/results/` holds cached grids, the run `20261001-234247_ar_K20_L3_kappa1e+08/`, `runs.csv` and `events_ar_K20_L3.csv`.


## Status (2026-10-02)
| Section | Status |
|---|---|
| §0 survey, adapter, session index, QC | done |
| §2 library (ssm), §3 features, §4 splits | done |
| §7 synthetic checks | done, all pass |
| §5 timescale, κ, M1, M2 grid + LOAO | done; K and κ selection unresolved (see Results) |
| §6 validation (1–4, 6) | done for the provisional K=20, L=3, κ=1e8; rerun at the chosen K/κ |
| M3 HSMM | justified by §6.3, not yet run |
| §6.5 neural predictivity | deferred, no neural data |

See [info.md](info.md) for how to run the code, the learning path and references.

## Next steps (current methods)
1. **Choose K and κ** from the DECIDE plots. Favour the smallest K where held-out LL levels off for the held-out animals *and* most states reproduce (Jaccard > 0.75). Favour a κ where the decoded and implied dwell times roughly agree.
2. **Rerun FINAL → DESCRIBE** at that setting. Set `K_sel`, `L_sel` and `kappa` in the cells, or add them to the config.
3. **M3:** HSMM vs sticky AR-HMM on the same LOSO folds. If the HSMM wins, rerun REPRO / GENERATIVE / BOUNDARIES with it, since its dwell times are modelled rather than implied.
4. **Name the states** from the state-triggered averages and the event table: pull 10–20 clips per state from video. Merge mirror pairs for description.
5. **Robustness:**
   - rerun with `observations="robust_ar"` (Student-t noise; tolerant of residual glitches);
   - vary `fs_out` (30 → 60 Hz) and `sg_win_s`;
   - check that the states survive.
6. **Development axis:** once states are fixed, add the EO < 8 sessions as *test* data (decode only), and compare state occupancy, dwell times and transitions across EO.
7. **Neural predictivity (§6.5)** when recordings are available.

## Alternatives, if these methods are not adequate

### (1) For identifying movement states
**Symptoms that call for a change:**
- states don't reproduce across restarts or halves, even at small K;
- held-out LL never levels off;
- dwell times misfit even with the HSMM;
- states have no consistent kinematic signature in the triggered averages and video.

**Data and features first** (the cheapest fixes):
- **More data.** The current cohort is 41 min from 2 animals. Add ferrets and sessions (including EO < 8) so LOAO has more than 2 folds and large K is supported.
- **Richer features:**
  - ω_rel (head yaw rate minus the rotation of the travel direction), which separates head-on-body turns from whole-body turns;
  - forward and lateral velocity in the head frame;
  - pitch and roll velocity;
  - z / rearing;
  - eye or gaze velocity, from the existing eye data.
- **Rate:** fit at 60 Hz, or on 120 Hz data with lag-stacked features, if fast movements are smeared at 30 Hz.

**Model alternatives, roughly in order of effort:**
- **Sticky HDP-AR-HMM / keypoint-MoSeq** (`jax-moseq`; Wiltschko et al. 2015, Weinreb et al. 2024).
  - Gibbs sampling, and the data pick the number of states used.
  - κ is set by a target median syllable duration, the same idea as here, but with full Bayesian uncertainty.
  - This is the standard tool for "syllables" from pose or kinematics.
- **HDP-HSMM** (`pyhsmm`; Johnson & Willsky 2013): nonparametric with explicit duration distributions.
- **Recurrent switching LDS (rSLDS)** (`ssm.SLDS` / `dynamax`; Linderman et al. 2017).
  - A continuous low-dimensional latent state with discrete switches that depend on where the latent is.
  - Better when movements are a continuum with a few regimes rather than discrete motifs.
- **Hierarchical / two-timescale models:** slow behavioural modes (explore, rest, hunt) that each contain fast movement motifs. Fit an HMM on windowed state-usage vectors from the AR-HMM, or use a hierarchical HMM.
- **Input-driven HMM (GLM-HMM)** (Ashwood et al. 2022): if task or stimulus variables exist (e.g. prey position during hunting), transitions can depend on them.
- **Non-Markov embeddings:**
  - MotionMapper (Berman et al. 2014): wavelet spectrogram → t-SNE/UMAP → watershed;
  - B-SOiD (Hsu & Yttri 2021);
  - VAME (Luxem et al. 2022): an RNN autoencoder with an HMM on the latent.
  - Useful as an independent check: states found by both approaches are more credible.
- **Clustering of segments:** PELT segments (already computed) → per-segment summaries → GMM / hierarchical clustering. Simple, transparent, and a good sanity baseline.

### (2) For identifying head turns (the original goal)
The HMM gives turn *states*, not turn *events*. Ways to get from states to turns, or to detect turns directly:
- **From the HMM:**
  - define a turn as a contiguous run of the mirror-paired turn states (merge adjacent turn states);
  - take onset and offset from the posterior crossing 0.5 rather than Viterbi;
  - report the amplitude (∫ω dt), peak velocity and duration of each turn.
- **Benchmark against the existing detector.**
  - `utils/process_session.py` already detects head saccades with a velocity threshold: `extract_saccades(R, 'skull', …)` → `df_head`.
  - Compare HMM turns with `df_head`: event-level precision and recall, and onset-time differences.
  - Agreement validates both; disagreements are the interesting cases to check on video.
- **Main-sequence validation** (`analyses/main_sequence.py`): real head turns should show a stereotyped amplitude–peak velocity–duration relation. Turns that fall off it are likely mis-segmented.
- **Eye–head coordination** (`utils/eye_head_timing.py`):
  - gaze-shifting turns should coincide with eye saccades;
  - compensatory movements should show VOR-like eye counter-rotation.
  - Use this to split turn types (gaze shifts vs other head movements) without hand labels.
- **Dedicated turn models**, if the general model blurs turns:
  - a 3-state HSMM on ω_yaw only (still / left / right) at 60–120 Hz;
  - a left-to-right "turn template" HMM (onset → accelerate → decelerate → settle) embedded in a background state;
  - changepoint detection on ω_yaw alone at full rate.
- **Parametric velocity profiles:** fit a minimum-jerk or gamma-shaped velocity pulse to each candidate turn. The fit quality and parameters (amplitude, duration, asymmetry) give both a detection criterion and a description.
- **Head-on-body vs whole-body turns:** use ω_rel and travel-direction change to separate turns of the head relative to the body from turns of the whole animal. This matters for interpreting gaze.
- **Light supervision, as a last resort:**
  - label a small set of turns in video and use it *only to evaluate* (precision and recall of each method);
  - if unsupervised methods remain inadequate, train a classifier on window features (e.g. A-SOiD active learning; Tillmann et al. 2024).
  - This departs from the unsupervised principle in §1, so treat it as a separate model.

## Feature change (2026-10-02): no extra filtering
- The skull kinematics are already smoothed upstream by the rigid-body fit; the loaded signals have under 1% of their power above 15 Hz. The non-smooth moments that remain are real.
- **Decision:** use the loaded data exactly as is. ω_yaw = `np.gradient` of loaded yaw, with no Savitzky–Golay and no low-pass (`sg_win_s: null`, `lowpass_hz: null`). Resampling to 30 Hz then aliases under 1% of the power.
- All earlier fits (everything in "Results, first full run" above, plus the DECIDE plots) used SG 100 ms + 12 Hz low-pass. They are archived in `HMM/results_filtered_sg100ms_lp12/` and are not comparable with new fits.
- K=4, L=3, κ=100 (SELECTED cell) was chosen on the filtered features. Rerun M1 / M2 / DECIDE on the unfiltered features to confirm it.

## K=4 run on unfiltered features (2026-10-02; 3 features, archived in `HMM/results_3feat_unfiltered/`)
- **What the states were:** 4 speed regimes, not turns.
  - still/slow, log speed 2.4: 21% of frames
  - moderate, 3.3: 18%
  - fast sustained, 4.8: 52%
  - fast burst with head down, 4.8: 9%
- **Reproducibility:** ARI across restarts 0.999; across data halves 0.47 (vs 0.69 on filtered features).
- **Dwell times:** the model reproduces them (KS ≤ 0.13), so an HSMM is not needed.
- **Boundaries:** 7.1% fall within ±100 ms of a changepoint, against 4.7% by chance.
- **Turn direction is missing:**
  - every state is its own mirror partner;
  - the ω_yaw averages around state onsets are flat;
  - the model's ω_yaw spectrum is 3–10× too low above about 5 Hz.
- **Decoded state duration** with unfiltered features at κ ≤ 1e3: about 0.1 s (vs 0.2 s filtered).

## Feature change (2026-10-02): head-frame velocity
- **Why:** to give the model a direct signal for turn direction. World x/y velocity would encode compass direction, so instead it is rotated into the head frame by the loaded yaw.
- **New features:**
  - `v_fwd` (forward/backward) and `v_lat` (sideways; flips sign under mirroring), each as signed log(|v| + 1);
  - features are now `[omega_yaw, log_speed, pitch, v_fwd, v_lat]` and `SIGNED = [0, 4]`;
  - no filtering.
- **Checks:**
  - `v_fwd > 0` in 89–95% of fast frames (speed > 200 mm/s), so the yaw and x/y conventions agree;
  - corr(ω_yaw, v_lat) ≈ 0.6, i.e. the head moves sideways in the direction of the turn.
- **Rerun from scratch:** M1, M2, LOAO, DECIDE K and DECIDE KAPPA. K/L/κ will be re-chosen from the new DECIDE plots; the SELECTED cell holds K=4, L=3, κ=100 only as a placeholder.
- **Compare with the 3-feature run on:**
  - whether mirror pairs appear (left/right turn states);
  - the ω_yaw state-triggered averages;
  - boundary agreement;
  - ARI across data halves.

## Results, 5 features (2026-10-02; run `HMM/results/20261002-112237_ar_K4_L3_kappa100`)
- **Selection:** K=4, L=3, κ=100 again.
  - M2 held-out LL rises to K=20 without plateau.
  - DECIDE K: K=4 is the largest K with restart ARI 1.0 and all states reproducible (K=6: 0.62, 17%).
  - DECIDE KAPPA: held-out LL flat for κ ≤ 1e3. Above that, implied dwell runs away from decoded.
- **States:**

  | State | Occupancy | Description | Mean dwell |
  |---|---|---|---|
  | 0 | 37% | forward locomotion (high speed and v_fwd) | 0.22 s |
  | 1, 2 | 20% each | left / right turn, a mirror pair. At onset, forward speed drops to about 0; then ω_yaw and v_lat peak and decay over about 0.3 s. | 0.21 s |
  | 3 | 25% | brief transitional state (speed and v_fwd drop at onset); hub between 0 and 1/2 | 0.10 s |

- **Validation:**
  - restart ARI 1.0; halves ARI 0.49;
  - dwell times are reproduced (KS ≤ 0.12), so no HSMM;
  - transitions are reproduced (r = 0.999);
  - boundary agreement with changepoints is weak (6.2% vs null 5.7%, p < 0.001);
  - the model's ω spectrum is still too low above about 5 Hz.
- **Takeaway:** the head-frame velocity gives turn-direction states, which the 3-feature model lacked.
- **Caveat:** states are short (median about 0.1 s), and a turn bout may span several visits.
- **Next:** compare turn states 1 and 2 with `extract_saccades` head events (`df_head`), and check them against video with `events_ar_K4_L3.csv`.

## K × κ sweep (2026-10-03; `HMM/results/sweep_K_kappa/`, script saved there)
- **Grid:** K ∈ {4, 5, 6} × κ ∈ {1e2, 1e3, 1e4}, L=3, 5 features.
- **κ:**
  - κ=1e2 and 1e3 give identical models.
  - **κ=1e4 breaks the turn pair** at K=4 and K=6, and gives only one turn direction at K=5. It also lowers restart and halves ARI, and lengthens states only from 0.10 to 0.13 s (median).
- **K, at κ ≤ 1e3:** every K keeps the left/right turn pair.
  - **K=5** adds a slow/still state (log speed 2.9) and a head-down state (pitch 0.25).
  - **K=6** also splits locomotion into moderate (3.5) and slow (2.2), plus head-down (0.18).
  - **Cost:** restart ARI falls from 1.0 (K=4) to 0.65 (K=5) and 0.62 (K=6); halves ARI falls from 0.49 to 0.38 and 0.33.
