I wrote binocularity_v2.py and it runs end-to-end (about 6 s), in both the normal and compare_delayed modes. It's committed in 7 steps; binocularity.py is unchanged.

Reorganization. It follows the vor_v2 layout:

one per-session table, with stats families data / saccade / slow / disconj / toy / pupil, each getting per-ferret r and a group LMM with Holm;
illustration cells after it: displacement scatters, disconjugacy vs amplitude, slow conjugacy vs head speed and locomotion with the LE-RE xcorr, saccade-triggered vergence, vergence vs toy distance, and pupil coupling.
Errors fixed (full list in the plan):

Delayed ferrets 403/411/416 were counted in the normal EO trend.
Six sessions with one eye essentially untracked were kept (402 EO6, 753 EO5/7/9, 757 EO2/3). Any session with both eyes tracked in under half the frames is now dropped.
axis="total" returned horizontal data. I removed it from helper_functions.py.
Saccade-free padding was 3/8 frames; it's now 12/20, as in vor_v2.
Error bars and r values pooled frames or saccades across sessions.
The amplitude sweep binned on the LE only and took an OLS slope that is biased low.
The CW/CCW labels were probably reversed; they're now yaw+ (left) / yaw− (right).
Some cells ignored the axis setting.
Two data problems found along the way, which changed how the measures are defined:

Horizontal and vertical are mixed in mirror image between the eyes. Each eye's vertical saccade displacement contains ±0.2–0.4 × the horizontal movement, with opposite signs in LE and RE, so raw LE-RE vertical r comes out negative. Vergence also changes by about −1.6 × the vertical displacement. This data can't tell a coordinate-frame artifact from real vergence linked to looking up or down. So horizontal measures use horizontal saccades only, and vertical conjugacy is a partial r with horizontal removed. The toy result is reported both raw and with vertical eye position removed. Both couplings are in the table as QC measures (xtalk, verg_vert).
The LE moves about 5–8% farther than the RE in both directions, varying by session (eye_ratio). This looks like a calibration mismatch, and it is why rightward saccades end converged and leftward ones diverged. The abducting/adducting measure averages the two directions so it cancels.
Results so far:

Saccade-free conjugacy rises with age: r_vel_x goes 0.10 → 0.16 → 0.20, LMM p = 0.006 (Holm), and rises in 5/6 ferrets. This only appeared after I added vor_v2's 36-frame tracking-gap margin, which landed in a vor_v2 commit made while I was working; before that it was noise.
Saccade conjugacy is high but dips in the oldest group: r_disp_x is 0.81 / 0.79 / 0.70, p = 0.045, driven mostly by F407 at EO ≥ 8. F407 EO11 also has the opposite cross-talk sign to every other session, so check its tracking.
Abducting eye is about 4% larger (Hering's-law asymmetry), with no EO trend.
No vergence change with toy distance in any group (r ≈ 0). Distance alone doesn't show the ferret is looking at the toy, so a gate on whether the toy is in front of the head comes next, once the yaw zero reference is checked.
Pupils are coupled between the eyes: peak r ≈ 0.3–0.4 at zero lag after correcting for eye position. There's no pupil–vergence coupling.
Changes from the plan:

The new measures are computed in the per-session table cell, as vor_v2 does, instead of each cell running its own stats.
The own-frame saccade table was dropped; the conjugate frame already tells which eye abducts.
The yaw comparison gates were loosened to |yaw| > 30 deg/s and tilt < 20°, because the planned 50 / 10 left only 2 of 42 sessions.
I added eye_ratio and the tracking-gap margin.
I saved these data problems to memory for later binocular or vertical work.