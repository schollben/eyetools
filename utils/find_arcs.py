import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils.session_data import SessionData


def find_arcs(session: SessionData, fs=120, max_gap_s=0.2, speed_min=50.0, min_dur_s=0.3, min_turn_deg=30,
              fine=dict(sigma_s=0.08, kappa_hi=0.015, kappa_lo=0.01125, merge_gap_s=0.0, edge_frac=0.0),
              coarse=dict(sigma_s=0.24, kappa_hi=0.001, kappa_lo=0.00075, merge_gap_s=0.4, edge_frac=0.33),
              plot=False, plot_window_s=(0, 60)):
    """Arcs (sustained same-direction turns) in the head's 2D path, at two smoothing scales.

    Fine scale: tight arcs with precise edges. Coarse scale: large arcs (radius up to ~1000 mm),
    which only rise above the head's side-to-side wiggle after heavy smoothing; added where no
    fine arc of the same direction is. Per scale: runs where |curvature| > kappa_lo that reach
    kappa_hi (speed > speed_min), edges pulled in to edge_frac x the run's peak, same-direction
    runs joined across gaps <= merge_gap_s, then kept if >= min_dur_s and turning >= min_turn_deg.
    Defaults were tuned on realistic synthetic paths (analyses/test_findarcs.py).

    Returns one row per arc: onset, end (exclusive frame), direction ('left' = CCW), scale,
    duration_s, turn_deg (+ = left), radius_mm (path length / turn), mean_speed_mm_s, path_length_mm.
    plot=True draws the trajectory and speed / curvature traces over plot_window_s.
    """
    vv = np.hypot(session.linearVel_x, session.linearVel_y)
    bad = ~(vv < 800)                                    # same head-speed mask as process_session
    x_raw = np.where(bad, np.nan, session.position_x).astype(float)
    y_raw = np.where(bad, np.nan, session.position_y).astype(float)
    n = len(x_raw)

    def runs(m):
        d = np.diff(m.astype(int), prepend=0, append=0)
        return np.where(d == 1)[0], np.where(d == -1)[0]

    def smooth(v, sigma_s):
        # fill gaps <= max_gap_s linearly, Gaussian-smooth, restore longer gaps as NaN
        bad_v = np.isnan(v)
        idx = np.arange(n)
        f = np.interp(idx, idx[~bad_v], v[~bad_v])
        for s, e in zip(*runs(bad_v)):
            if e - s > max_gap_s * fs:
                f[s:e] = np.nan
        out = gaussian_filter1d(np.nan_to_num(f, nan=np.nanmean(f)), sigma_s * fs)
        out[np.isnan(f)] = np.nan
        return out

    def detect(p):
        x, y = smooth(x_raw, p['sigma_s']), smooth(y_raw, p['sigma_s'])
        dx, dy = np.gradient(x) * fs, np.gradient(y) * fs
        ddx, ddy = np.gradient(dx) * fs, np.gradient(dy) * fs
        speed = np.hypot(dx, dy)
        kappa = (dx * ddy - dy * ddx) / np.maximum(speed, 1e-9) ** 3      # signed: + = left (CCW)
        moving = speed > speed_min                                         # NaN compares False
        segs = []
        for sign in (1, -1):
            k = sign * kappa
            hi = moving & (k > p['kappa_hi'])
            kept = []
            for s0, e0 in zip(*runs(moving & (k > p['kappa_lo']))):
                if not hi[s0:e0].any():
                    continue
                thr = max(p['kappa_lo'], p['edge_frac'] * k[s0:e0].max())
                for s, e in zip(*runs(k[s0:e0] > thr)):
                    s, e = s + s0, e + s0
                    if not hi[s:e].any():
                        continue
                    if kept and s - kept[-1][1] <= p['merge_gap_s'] * fs:
                        kept[-1] = (kept[-1][0], e)
                    else:
                        kept.append((s, e))
            segs += [(s, e, sign) for s, e in kept if (e - s) / fs >= min_dur_s
                     and np.rad2deg(np.nansum(k[s:e] * speed[s:e]) / fs) >= min_turn_deg]
        return segs, (x, y, speed, kappa)

    segs_f, K_f = detect(fine)
    segs_c, K_c = detect(coarse)
    in_fine = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    for s, e, sign in segs_f:
        in_fine[sign][s:e] = True
    segs = sorted([(s, e, sign, 'fine') for s, e, sign in segs_f]
                  + [(s, e, sign, 'coarse') for s, e, sign in segs_c if not in_fine[sign][s:e].any()])

    rows = []
    for s, e, sign, scale in segs:
        _, _, speed, kappa = K_f if scale == 'fine' else K_c
        turn = np.nansum(kappa[s:e] * speed[s:e]) / fs
        length = np.nansum(speed[s:e]) / fs
        rows.append(dict(onset=s, end=e, direction='left' if sign > 0 else 'right', scale=scale,
                         duration_s=(e - s) / fs, turn_deg=np.rad2deg(turn),
                         radius_mm=length / max(abs(turn), 1e-9),
                         mean_speed_mm_s=np.nanmean(speed[s:e]), path_length_mm=length))
    df = pd.DataFrame(rows, columns=['onset', 'end', 'direction', 'scale', 'duration_s', 'turn_deg', 'radius_mm',
                                     'mean_speed_mm_s', 'path_length_mm'])

    if plot:
        left_c, right_c, coarse_c = '#2a78d6', '#eb6834', '#8a8984'
        x, y, speed, kappa = K_f
        t = np.arange(n) / fs
        w0, w1 = int(plot_window_s[0] * fs), int(plot_window_s[1] * fs)
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], hspace=0.3)
        ax = fig.add_subplot(gs[:, 0])
        ax.plot(x_raw[w0:w1], y_raw[w0:w1], '.', ms=1, color='#b9b8b2', label='raw')
        ax.plot(x[w0:w1], y[w0:w1], color='#0b0b0b', lw=0.8, label='smoothed (fine)')
        for s, e, sign, scale in segs:
            if e > w0 and s < w1:
                ax.plot(x[s:e], y[s:e], lw=4 if scale == 'fine' else 8, alpha=0.6 if scale == 'fine' else 0.3,
                        color=left_c if sign > 0 else right_c)
        ax.plot([], [], lw=4, color=left_c, label='left arc'); ax.plot([], [], lw=4, color=right_c, label='right arc')
        ax.plot([], [], lw=8, alpha=0.3, color='0.4', label='coarse-scale arc')
        ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(loc='best', fontsize=7)
        ax.set_title(f'{session.id} EO{session.eo}: {len(df)} arcs')

        a1 = fig.add_subplot(gs[0, 1])
        a2 = fig.add_subplot(gs[1, 1], sharex=a1)
        a3 = fig.add_subplot(gs[2, 1], sharex=a1)
        a1.plot(t, speed, color='#0b0b0b', lw=1); a1.axhline(speed_min, color='#e34948', ls='--', lw=1)
        a1.set_ylabel('speed (mm/s)')
        for (_, _, sp, k), p, c, c_thr in ((K_f, fine, '#0b0b0b', '#e34948'), (K_c, coarse, coarse_c, coarse_c)):
            a2.semilogy(t, np.where(sp > speed_min, np.abs(k), np.nan), color=c, lw=1)
            a2.axhline(p['kappa_hi'], color=c_thr, ls='--', lw=1); a2.axhline(p['kappa_lo'], color=c_thr, ls=':', lw=1)
            a3.plot(t, np.where(sp > speed_min, k, np.nan), color=c, lw=1)
        a2.set_ylabel('|curvature| (1/mm)')
        a3.axhline(0, color=coarse_c, lw=0.6)
        a3.set_ylabel('curvature (+ left)'); a3.set_xlabel('time (s)')
        for a in (a1, a2, a3):
            for s, e, sign, scale in segs:
                a.axvspan(s / fs, e / fs, color=left_c if sign > 0 else right_c,
                          alpha=0.15 if scale == 'fine' else 0.07, lw=0)
            a.spines[['top', 'right']].set_visible(False)
        a1.set_xlim(plot_window_s)

    return df
