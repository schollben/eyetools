# VOR analysis — potential to-do

## Measurement

- [ ] Signed per-axis VOR: `yaw_v` vs eye `vx`, `pitch_v` vs eye `vy` (replace or add to speed-vs-speed, cells 1–2)
- [ ] Axis alignment check: regress `yaw_v` against both `vx` and `vy`; the cross-term should be ~0

## Fitting

- [ ] Unbiased gain: total least squares or binned medians (OLS attenuates the slope when head velocity is noisy)
- [ ] Per-session / per-eye gain, then aggregate (instead of pooling all frames)
- [ ] Exclude low head velocity (|head vel| < noise floor, ~20–30 deg/s) or report gain per head-velocity bin

## Timing

- [ ] Latency: find the lag with `run_xcorr` and shift eye vs head before fitting
- [ ] Filter head and eye velocity the same way

## Frame selection

- [ ] Longer `pad_post` (e.g. 30, as in integrator.py); compare non-saccade eye-velocity tails at 12 vs 30
- [ ] Remove missed quick phases with a residual threshold (|eye vel + gain·head vel|), NOT a raw eye-velocity ceiling
- [ ] Gain vs frequency: band-pass (<1, 1–4, >4 Hz) or cross-spectral gain and coherence
- [ ] Gain vs eye position bin (links to integrator results)
- [ ] Running vs stationary gain (`loco_subset`)

## Controls

- [ ] Shuffle control: circularly shift eye vs head for the null gain and r²
- [ ] Binocular consistency: compare LE vs RE yaw gain

Priority: signed per-axis, unbiased fit, low-velocity exclusion, latency.
