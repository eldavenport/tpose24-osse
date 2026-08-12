"""Manuscript figures for the orientation-sensitivity note (physics audience)."""
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle

import test_footprint_pipeline as T

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.labelsize': 10, 'axes.titlesize': 10.5, 'font.size': 9.5,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.6, 'figure.dpi': 200,
})

COL = {'triangle': '#E69F00', 'square': '#D55E00', 'pentagon': '#009E73',
       'hexagon': '#0072B2', 'octagon': '#CC79A7', '12-gon': '#56B4E9'}
RAD = 0.75  # glider radius r (deg)


def ngon(n, r=RAD, rot0=np.pi / 2):
    a = rot0 + 2 * np.pi * np.arange(n) / n
    return [(float(r * np.sin(t)), float(r * np.cos(t))) for t in a]  # (dlat,dlon)


NGONS = {'triangle': ngon(3), 'square': ngon(4), 'pentagon': ngon(5),
         'hexagon': ngon(6), 'octagon': ngon(8)}


# ---------------------------------------------------------------- Figure 1
def fig_polygons():
    order = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    disp = {'triangle': ngon(3), 'square': ngon(4, rot0=np.pi / 4),
            'pentagon': ngon(5), 'hexagon': ngon(6), 'octagon': ngon(8, rot0=np.pi / 8)}
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.95))
    for ax, nm in zip(axs, order):
        n = len(disp[nm])
        xy = np.array([[o[1], o[0]] for o in disp[nm]])
        ax.add_patch(MplPoly(xy, closed=True, fill=True, fc=COL[nm], ec='k', lw=1.2, alpha=0.22))
        ax.scatter(xy[:, 0], xy[:, 1], s=34, c=COL[nm], ec='k', lw=0.8, zorder=5)
        ax.set_title(f'{nm}  ($n={n}$)', fontsize=9.5)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.15, 1.05)
        ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
    fn = os.path.join(HERE, 'fig1_polygons.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 2
def fig_aliasing():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5))
    grid = np.linspace(-1.18, 1.18, 260)
    GX, GY = np.meshgrid(grid, grid)
    PHI = np.arctan2(GY, GX); RR = np.hypot(GX, GY)
    for ax, (m, alias) in zip(axs, [(3, 'triangle'), (4, 'square')]):
        Z = np.cos(m * PHI)
        Z = np.ma.masked_where(RR > 1.0, Z)
        ax.pcolormesh(grid, grid, Z, cmap='RdBu_r', vmin=-1, vmax=1,
                      shading='auto', rasterized=True)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec='k', lw=1.2))
        vx = np.cos(2 * np.pi * np.arange(m) / m)   # vertices on the peaks (worst alignment)
        vy = np.sin(2 * np.pi * np.arange(m) / m)
        ax.add_patch(MplPoly(np.c_[vx, vy], closed=True, fill=False, ec='k', lw=1.6, zorder=4))
        ax.scatter(vx, vy, s=95, c='#111', ec='w', lw=1.2, zorder=6)
        ax.set_title(rf'{m}-lobed flow signature  (mode $m={m}$)', fontsize=9.8)
        ax.text(0, -1.34, rf'{alias} ($n={m}$): all $n$ gliders on the same phase'
                '\n$\\Rightarrow$ pattern averages to a constant (aliased)',
                ha='center', va='top', fontsize=8.4)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.6, 1.2)
        ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.06)
    fn = os.path.join(HERE, 'fig6_aliasing.pdf')
    fig.savefig(fn, bbox_inches='tight', dpi=150); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 4
def fig_rotation_curves():
    # names/formulas match Table 1; no mode labels -- this figure precedes the mechanism
    cases = [('even', 'curvature flow\n$U=C\\,x^{2}$'),
             ('odd', 'cubic flow\n$U=C\\,x^{3}$'),
             ('mixed', 'cubic + cross-term\n$U=C\\,(x y^{2}+\\frac{1}{2}x^{3})$')]
    order = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    th = np.arange(0, 180.5, 1.5)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True)
    for ax, (case, ttl) in zip(axs, cases):
        for nm in order:
            de = np.array([T._rot_planefit_div(case, NGONS[nm], d) for d in th]) * 1e7
            de = de - de.mean()
            lw = 2.4 if (de.max() - de.min()) > 0.05 else 1.2
            ax.plot(th, de, color=COL[nm], lw=lw, label=nm)
        ax.axhline(0, color='0.6', lw=0.7, ls=':')
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(r'flow rotation $\theta$ (deg)')
        ax.set_xticks([0, 45, 90, 135, 180])
    axs[0].set_ylabel(r'divergence-estimate swing''\n'r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    axs[1].legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.46),
                  frameon=False, columnspacing=1.1, handlelength=1.5)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.78, bottom=0.16, wspace=0.24)
    fn = os.path.join(HERE, 'fig3_rotation_curves.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 3 (worked example)
def fig_example():
    """One concrete flow end-to-end: the velocity field, its true divergence, and how the
    plane-fit estimate of the two candidate OSSE arrays swings as the flow is rotated."""
    import osse_tools as ot
    W = T.WIDTH
    Wh, Hx = T._shape_box('hexagon')
    sq = ot.footprint_offsets('square4', W, W)
    hx = ot.footprint_offsets('hexagon', Wh, Hx)
    sqxy = np.array([[o[1], o[0]] for o in sq])
    hxxy = np.array([[o[1], o[0]] for o in hx])
    sq_out = np.array([[p[1], p[0]] for p in ot.footprint_outline('square4', W, W)])
    hx_out = np.array([[p[1], p[0]] for p in ot.footprint_outline('hexagon', Wh, Hx)])

    half = 1.0
    g = np.linspace(-half, half, 240); GX, GY = np.meshgrid(g, g)
    D = T._rot_div('combo', GX, GY, 0.0) * 1e6
    qs = np.linspace(-0.85, 0.85, 12); QX, QY = np.meshgrid(qs, qs)
    QU, QV = T._rot_uv('combo', QX, QY, 0.0)

    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.75),
                            gridspec_kw={'width_ratios': [1.0, 1.0, 1.3]})

    def _arrays(ax):
        ax.add_patch(MplPoly(sq_out, closed=True, fill=False, ec='#D55E00', lw=1.3, zorder=4))
        ax.add_patch(MplPoly(hx_out, closed=True, fill=False, ec='#0072B2', lw=1.3, zorder=4))
        ax.scatter(sqxy[:, 0], sqxy[:, 1], marker='s', s=30, c='#D55E00', ec='k', lw=0.5, zorder=6)
        ax.scatter(hxxy[:, 0], hxxy[:, 1], marker='o', s=30, c='#0072B2', ec='k', lw=0.5, zorder=6)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    # (a) velocity
    axs[0].quiver(QX, QY, QU, QV, color='#333', scale=1.5, width=0.006)
    _arrays(axs[0])
    axs[0].set_title('(a) the flow (velocity)', fontsize=9.2)

    # (b) divergence
    dm = float(np.nanmax(np.abs(D)))
    im = axs[1].pcolormesh(g, g, D, cmap='PuOr_r', vmin=-dm, vmax=dm, shading='auto',
                           rasterized=True)
    _arrays(axs[1])
    axs[1].set_title('(b) its divergence', fontsize=9.2)
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
    cb.ax.set_title(r'$10^{-6}\,\mathrm{s}^{-1}$', fontsize=7.2, pad=4)
    cb.ax.tick_params(labelsize=7)

    # (c) estimate error vs rotation
    th = np.arange(0, 180.5, 3)
    err_sq = np.array([T._rot_planefit_div('combo', sq, d)
                       - T._rot_truth('combo', 'square4', d, W, W, nfine=161) for d in th]) * 1e7
    err_hx = np.array([T._rot_planefit_div('combo', hx, d)
                       - T._rot_truth('combo', 'hexagon', d, Wh, Hx, nfine=161) for d in th]) * 1e7
    axs[2].plot(th, err_sq, color='#D55E00', lw=2.0, label='square, $n=4$')
    axs[2].plot(th, err_hx, color='#0072B2', lw=2.0, label='hexagon, $n=6$')
    axs[2].axhline(0, color='0.55', lw=0.8, ls=':')
    axs[2].set_title('(c) estimate error vs orientation', fontsize=9.2)
    axs[2].set_xlabel(r'flow rotation $\theta$ (deg)')
    axs[2].set_ylabel(r'error $(\times10^{-7}\,\mathrm{s}^{-1})$')
    axs[2].set_xticks([0, 45, 90, 135, 180])
    lo, hi = min(err_sq.min(), err_hx.min()), max(err_sq.max(), err_hx.max())
    axs[2].set_ylim(lo - 0.10 * (hi - lo), hi + 0.42 * (hi - lo))   # headroom for the legend
    axs[2].legend(frameon=False, fontsize=7.8, loc='upper center', ncol=2,
                  columnspacing=1.0, handlelength=1.4)
    axs[2].grid(True, alpha=0.25)

    # one shared array legend under panels a/b
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor='#D55E00',
                      markeredgecolor='k', label='square array'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072B2',
                      markeredgecolor='k', label='hexagon array')]
    axs[0].legend(handles=handles, frameon=False, fontsize=7.6, loc='lower center',
                  bbox_to_anchor=(0.5, -0.24), ncol=2, columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.9, bottom=0.16, wspace=0.45)
    fn = os.path.join(HERE, 'fig2_example.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 5
def _disc_rms(case, r=RAD, n=201):
    """RMS speed of a test flow inside the array disc -- used to put the wave field and the
    centered feature on a common amplitude footing."""
    g = np.linspace(-r, r, n); GX, GY = np.meshgrid(g, g)
    m = np.hypot(GX, GY) <= r
    U, V = T._rot_uv(case, GX, GY, 0.0)
    return float(np.sqrt(np.mean(U[m]**2 + V[m]**2)))


def fig_features():
    """Broadband waves, with and without a centered (non-reversing) feature.

    Left: absolute orientation error per shape -- the feature moves ONLY the triangle, but
    the square is the loudest shape on the waves alone. Right: the same as a function of the
    feature's strength -- four flat lines, one rising, so which shape is worst depends on the
    flow, while WHICH shape responds to the feature does not."""
    order = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    th = np.arange(0, 360, 2.0)

    def curve(case, nm):                       # estimate swing about its own mean, 1e-7 /s
        de = np.array([T._rot_planefit_div(case, NGONS[nm], d) for d in th]) * 1e7
        return de - de.mean()

    k = _disc_rms('combo') / _disc_rms('even')   # 'strength 1' = same RMS speed as the waves
    wave = {nm: curve('combo', nm) for nm in order}
    feat = {nm: k * curve('even', nm) for nm in order}
    e_w = np.array([np.ptp(wave[nm]) for nm in order])
    e_b = np.array([np.ptp(wave[nm] + feat[nm]) for nm in order])
    print('  waves only      :', dict(zip(order, e_w.round(2))))
    print('  waves + feature :', dict(zip(order, e_b.round(2))))

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2),
                            gridspec_kw={'width_ratios': [1.18, 1.0]})

    # (a) absolute error, with and without the centered feature
    ax = axs[0]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, e_w, w, color='#7FA8C9', ec='k', lw=0.4, label='waves only')
    ax.bar(x + w / 2, e_b, w, color='#B24C3C', ec='k', lw=0.4,
           label='waves + centered feature')
    ax.annotate('', xy=(x[0] - w / 2, e_b[0] * 1.005), xytext=(x[0] - w / 2, e_w[0]),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#7a2f22',
                                shrinkA=0, shrinkB=0))
    ax.text(x[0] + 0.42, e_b[0] * 1.03, 'only the triangle\nresponds', ha='left', va='bottom',
            fontsize=8.2, color='#7a2f22', fontweight='bold')
    ax.text(3.0, e_b[0] * 0.55, 'all other shapes:\nunchanged',
            ha='center', va='center', fontsize=8.2, color='0.35')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nm}\n($n={len(NGONS[nm])}$)' for nm in order], fontsize=8.2)
    ax.set_ylabel('orientation error, peak-to-peak\n'
                  r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    ax.set_title('(a) with and without a centered feature', fontsize=9.4)
    ax.set_ylim(0, max(e_b) * 1.42)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.30),
              ncol=2, fontsize=8.4, columnspacing=1.2, handlelength=1.4)

    # (b) sweep the feature's strength: four flat lines, one rising
    ax = axs[1]
    al = np.linspace(0, 2.0, 81)
    for nm in order:
        tot = np.array([np.ptp(wave[nm] + a * feat[nm]) for a in al])
        ax.plot(al, tot, color=COL[nm], lw=2.2 if nm == 'triangle' else 1.5, label=nm)
    ax.axvline(1.0, color='0.55', lw=0.9, ls='--')
    ax.text(0.96, ax.get_ylim()[1] * 0.97, 'panel (a)', fontsize=7.8, color='0.4',
            rotation=90, ha='right', va='top')
    ax.set_xlabel('strength of the centered feature\n(RMS speed relative to the waves)')
    ax.set_ylabel('orientation error, peak-to-peak\n'
                  r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    ax.set_title('(b) sensitivity to the feature strength', fontsize=9.4)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.30),
              ncol=5, fontsize=8.2, columnspacing=0.9, handlelength=1.2)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.76, bottom=0.20, wspace=0.42)
    fn = os.path.join(HERE, 'fig4_features.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 6
def _swing_curve(offsets, x0, y0, th):
    """Peak-to-peak swing of the plane-fit divergence over a full flow rotation, for the
    'combo' wave field whose origin sits at (-x0, -y0) relative to the array center. A
    centered array (x0=y0=0) straddles a zero crossing and sees a purely even-lobed field;
    any offset breaks that symmetry. Vectorized over rotation angle."""
    import osse_tools as ot
    o = np.asarray(offsets, float)
    t = np.radians(th)[:, None]; cs, sn = np.cos(t), np.sin(t)
    x, y = o[None, :, 1], o[None, :, 0]
    xp, yp = x * cs + y * sn, -x * sn + y * cs
    up, vp = T._UV('combo', xp + x0, yp + y0)
    U, V = up * cs - vp * sn, up * sn + vp * cs
    wx, wy = ot._planefit_slope_weights(offsets, T.LAT0)
    c = (U @ wx + V @ wy) * 1e7
    return float(np.ptp(c - c.mean()))


def fig_ensemble(nplace=400, seed=0, half=1.5):
    """The general case: the same wave field sampled at many array placements. The special
    centered placement (stars) is what Figs 4-5 use; averaged over placements the ordering is
    monotone in n and the pentagon's apparent advantage disappears."""
    order = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    th = np.arange(0, 360, 4.0)
    rng = np.random.default_rng(seed)
    places = rng.uniform(-half, half, size=(nplace, 2))
    data = {nm: np.array([_swing_curve(NGONS[nm], x0, y0, th) for x0, y0 in places])
            for nm in order}
    cen = {nm: _swing_curve(NGONS[nm], 0.0, 0.0, th) for nm in order}
    print('  ensemble means :', {nm: round(float(data[nm].mean()), 2) for nm in order})

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bp = ax.boxplot([data[nm] for nm in order], positions=x, widths=0.55, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='k', lw=1.4))
    for patch, nm in zip(bp['boxes'], order):
        patch.set_facecolor(COL[nm]); patch.set_alpha(0.45); patch.set_edgecolor('k')
    ax.plot(x, [np.mean(data[nm]) for nm in order], color='0.35', lw=1.1, ls='--',
            marker='o', ms=4, zorder=5, label='mean over placements')
    ax.plot(x, [cen[nm] for nm in order], ls='none', marker='*', ms=14, mfc='w',
            mec='#B24C3C', mew=1.6, zorder=6,
            label='the one centered placement (Figs. 3 and 4)')
    ax.annotate('a lucky placement, not a better shape',
                xy=(x[2], cen['pentagon'] - 0.35), xytext=(x[2], -1.35),
                fontsize=8.2, color='#B24C3C', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#B24C3C', lw=1.1))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nm}\n($n={len(NGONS[nm])}$)' for nm in order], fontsize=8.4)
    ax.set_ylabel('orientation error, peak-to-peak\n'
                  r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    top = ax.get_ylim()[1]
    ax.set_ylim(-2.1, top)                        # empty strip below zero for the annotation
    ax.set_yticks(np.arange(0, top, 2.0))
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=2,
              fontsize=8.4, columnspacing=1.4, handlelength=1.6, numpoints=1)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.86, bottom=0.15)
    fn = os.path.join(HERE, 'fig5_ensemble.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


# ---------------------------------------------------------------- Figure 7
OFFSET = (0.37, 0.21)   # deg; the off-center placement used as the concrete alternative case


def _g_spectrum(x0, y0, n=512):
    """Azimuthal amplitude spectrum |g_m| of the scalar the plane fit actually sums,
    g(phi) = x U + y V around the glider ring, expressed directly in swing units: an array
    whose lowest blind mode is m suffers a peak-to-peak swing of 8|g_m|/r^2. So the value
    read off at m=n IS that n-gon's predicted orientation error."""
    ph = np.arange(n) * 2 * np.pi / n
    x, y = RAD * np.cos(ph), RAD * np.sin(ph)
    U, V = T._UV('combo', x + x0, y + y0)
    g = (x * T.M_PER_DEG) * U + (y * T.M_PER_DEG) * V
    G = np.abs(np.fft.fft(g) / n)
    return 8.0 * G / (RAD * T.M_PER_DEG) ** 2 * 1e7


def fig_spectrum(mmax=12):
    """Why parity decides the ranking. (a) What the array samples, by azimuthal wavenumber:
    centered on the wave field it has NO odd content, so the odd-n arrays are blind at a
    wavenumber that carries nothing. (b) The concrete consequence, centered vs off-center."""
    order = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    ns = [len(NGONS[nm]) for nm in order]
    Sc, So = _g_spectrum(0.0, 0.0), _g_spectrum(*OFFSET)
    m = np.arange(1, mmax + 1)
    th = np.arange(0, 360, 2.0)
    meas_c = np.array([_swing_curve(NGONS[nm], 0.0, 0.0, th) for nm in order])
    meas_o = np.array([_swing_curve(NGONS[nm], *OFFSET, th) for nm in order])
    print('  centered odd-m amplitudes:', np.round(Sc[1:10:2], 4))
    print('  offset   per-shape swing :', dict(zip(order, meas_o.round(2))))

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2),
                            gridspec_kw={'width_ratios': [1.28, 1.0]})

    # (a) the spectrum of what the array samples
    ax = axs[0]; w = 0.38
    ax.bar(m - w / 2, Sc[1:mmax + 1], w, color='#7FA8C9', ec='k', lw=0.4,
           label='centered on the field')
    ax.bar(m + w / 2, So[1:mmax + 1], w, color='#B24C3C', ec='k', lw=0.4,
           label=f'displaced by ({OFFSET[0]}$^\\circ$, {OFFSET[1]}$^\\circ$)')
    ax.set_xticks(m)
    ax.set_xlabel('azimuthal wavenumber $m$ of the sampled signal')
    ax.set_ylabel('swing an array blind at $m$ would suffer\n'
                  r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    ax.set_title('(a) what the array samples', fontsize=9.4)
    ax.grid(True, axis='y', alpha=0.25)
    top = ax.get_ylim()[1]
    ax.set_ylim(-0.22 * top, top)
    for nm, n in zip(order, ns):                 # each shape's lowest blind mode (colors = Fig. 1)
        ax.plot([n], [-0.11 * top], marker='^', ms=8, color=COL[nm], mec='k', mew=0.4,
                clip_on=False, zorder=6)
    # the markers are explained in the caption (colors keyed to Fig. 1)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_yticks(np.arange(0, top, 2.5))
    ax.text(6.4, 0.72 * top, 'centered: nothing at all\nat odd $m$ (1, 3, 5, 7, 9)',
            ha='left', va='center', fontsize=8.2, color='#3d6484')
    for mm in (3, 5):
        ax.annotate('', xy=(mm - w / 2, 0.015 * top), xytext=(6.3, 0.72 * top),
                    arrowprops=dict(arrowstyle='->', color='#3d6484', lw=0.9,
                                    connectionstyle='arc3,rad=0.22'))
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2,
              fontsize=8.2, columnspacing=1.2, handlelength=1.3)

    # (b) the consequence, shape by shape
    ax = axs[1]; x = np.arange(len(order))
    ax.bar(x - w / 2, meas_c, w, color='#7FA8C9', ec='k', lw=0.4)
    ax.bar(x + w / 2, meas_o, w, color='#B24C3C', ec='k', lw=0.4)
    def _active(S, n):
        """Lowest blind mode of an n-gon that actually carries signal (m = n, 2n, 3n, ...)."""
        for j in range(1, 6):
            if S[j * n] > 1e-3 * S[1:].max():
                return S[j * n]
        return 0.0
    ax.plot(x - w / 2, [_active(Sc, n) for n in ns], ls='none', marker='_', ms=9, mew=1.6,
            color='k', zorder=6, label='predicted from panel (a)')
    ax.plot(x + w / 2, [_active(So, n) for n in ns], ls='none', marker='_', ms=9, mew=1.6,
            color='k', zorder=6)
    ax.annotate('', xy=(x[2] + w / 2, meas_o[2] * 0.97), xytext=(x[2] - w / 2, meas_c[2] + 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#7a2f22',
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(x[2] + 0.06, meas_o[2] * 1.05, r'$0.15\rightarrow6.1$', ha='center', va='bottom',
            fontsize=8.2, color='#7a2f22', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nm}\n($n={n}$)' for nm, n in zip(order, ns)], fontsize=7.0)
    ax.set_ylabel('orientation error, peak-to-peak\n'
                  r'$(\times10^{-7}\,\mathrm{s}^{-1})$')
    ax.set_title('(b) the consequence', fontsize=9.4)
    ax.set_ylim(0, max(meas_c.max(), meas_o.max()) * 1.26)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, loc='upper right', fontsize=8.0, numpoints=1, handlelength=1.2)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.19, wspace=0.42)
    fn = os.path.join(HERE, 'fig7_spectrum.pdf')
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
    return fn


if __name__ == '__main__':
    print(fig_polygons())
    print(fig_aliasing())
    print(fig_example())
    print(fig_rotation_curves())
    print(fig_features())
    print(fig_ensemble())
    print(fig_spectrum())
    print('FIGS_DONE')
