"""
run_spectra.py — wavenumber-frequency spectra of the area-average w and the
advective heat tendency (w.dT/dz) for one representative equatorial cell.

The footprint average collapses horizontal space, so each quantity is a
(time, depth) series and the only "wavenumber" is the VERTICAL wavenumber m
(cpm); the other axis is frequency (cpd). We show:

  * spectra_2d_*.png  — 2D vertical-wavenumber-frequency spectra (log-log),
    rows = [area-mean w, advective heat tendency], columns = True, Estimate,
    and the estimate/true variance ratio.
  * spectra_1d_*.png  — depth-averaged 1D frequency spectra (log-log), true
    (solid) vs estimate (dashed) for both quantities.

True area-mean w = wbar_hull; estimated = w_est_mid (plane fit).
True heat tendency = A_true_total (footprint mean of w.dT/dz); estimated =
[w]*[dT/dz] = w_est_mid * d(Tbar_glider)/dz.

Data are 3-hourly over ~81 days (Nyquist 4 cpd) on a uniform 2 m grid,
9-79 m (vertical Nyquist 0.25 cpm).
"""

import os
import numpy as np
import xarray as xr
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, TwoSlopeNorm
import cmocean.cm as cmo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_heat")
OUT = os.path.join(HERE, "spectra_figs")
CONFIG = "equator_1deg_w0.5"
CELL = "+0.00"

# shift-array comparison: one figure per center latitude, 3 shapes overlaid/rowed
SHAPES = ["diamond", "hexagon", "square"]
SHAPE_COLOR = {"diamond": "#1f77b4", "hexagon": "#2ca02c", "square": "#d62728"}
SHIFT_BASE = {"diamond": "shift_w0.5", "hexagon": "shift_hex_w0.5",
              "square": "shift_sq_w0.5"}
SHIFT_LATS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

DT_DAY = 3.0 / 24.0          # 3-hourly sampling
FREQ_MAX = 1.0               # cap plots at 1 cpd (timescales > 1 day)
# --- presentation-ready defaults (repo convention) ---
plt.rcParams.update({
    "axes.labelsize": 12.5, "axes.labelweight": "bold",
    "legend.fontsize": 12, "legend.title_fontsize": 14,
    "axes.titlesize": 12.5,
})


def _prep(series2d):
    """Demean per depth and linearly detrend in time. series2d: (nt, nz)."""
    x = np.asarray(series2d, float)
    x = x - x.mean(axis=0, keepdims=True)
    x = signal.detrend(x, axis=0, type="linear")
    return x


def psd_1d(series2d, dt=DT_DAY):
    """Depth-averaged one-sided frequency PSD (Hann-tapered in time).

    Returns freq (cpd, >0) and PSD density averaged over depth.
    """
    x = _prep(series2d)
    nt, nz = x.shape
    win = np.hanning(nt)[:, None]
    xw = x * win
    F = np.fft.rfft(xw, axis=0)
    freq = np.fft.rfftfreq(nt, d=dt)
    df = freq[1] - freq[0]
    wcorr = (win[:, 0] ** 2).mean()            # window power
    psd = (np.abs(F) ** 2) / (nt ** 2) / df / wcorr
    psd[1:-1] *= 2.0                            # one-sided (fold negative freq)
    psd = psd.mean(axis=1)                      # depth average
    return freq[1:], psd[1:]                    # drop f=0 (log axis)


def psd_2d(series2d, dt=DT_DAY, dz=2.0):
    """Folded one-sided 2D (frequency, |vertical wavenumber|) PSD density.

    Hann-tapered in time and depth; power folded onto the positive quadrant so
    the integral over (f>0, m>0) recovers the field variance. Returns
    freq (cpd), kz (cpm), PSD[len(freq), len(kz)].
    """
    x = _prep(series2d)
    nt, nz = x.shape
    win = np.hanning(nt)[:, None] * np.hanning(nz)[None, :]
    xw = x * win
    F = np.fft.fft2(xw)
    freq = np.fft.fftfreq(nt, d=dt)
    kz = np.fft.fftfreq(nz, d=dz)
    df, dkz = 1.0 / (nt * dt), 1.0 / (nz * dz)
    wcorr = (win ** 2).mean()
    S = (np.abs(F) ** 2) / (nt * nz) ** 2 / (df * dkz) / wcorr
    # fold onto positive quadrant (real field: S(f,k)=S(-f,-k))
    fpos, kpos = np.unique(np.abs(freq)), np.unique(np.abs(kz))
    fi = np.searchsorted(fpos, np.abs(freq))
    ki = np.searchsorted(kpos, np.abs(kz))
    S1 = np.zeros((len(fpos), len(kpos)))
    FI = np.repeat(fi[:, None], nz, axis=1).ravel()
    KI = np.repeat(ki[None, :], nt, axis=0).ravel()
    np.add.at(S1, (FI, KI), S.ravel())
    return fpos[1:], kpos[1:], S1[1:, 1:]      # drop the zero rows/cols


def shift_path(shape, lat):
    """Path to the shift-config cell for a given shape and center latitude.
    Latitudes 0.0/±1.0 come from the `*_mid` configs, ±0.5/±1.5 from the base."""
    base = SHIFT_BASE[shape]
    cfg = base if round(abs(lat), 2) in (0.5, 1.5) else base + "_mid"
    return os.path.join(DATA, f"{cfg}__cell_{lat:+.2f}.nc")


def load_cell(path):
    ds = xr.open_dataset(path)
    z = np.asarray(ds["Z"])                    # negative-down (m)
    w_true = np.asarray(ds["wbar_hull"])       # (time, depth), m/day
    w_est = np.asarray(ds["w_est_mid"])
    A_true = np.asarray(ds["A_true_total"])    # deg C/day
    # array estimate of heating = [w]*[dT/dz], gradient from the glider-mean T
    Tg = np.asarray(ds["Tbar_glider"])
    dTdz = np.gradient(Tg, z, axis=1)          # deg C/m (z upward-positive)
    A_est = w_est * dTdz
    # true footprint-mean temperature + stratification for context (time, depth)
    T_true = np.asarray(ds["Tbar_hull"])
    dTdz_true = np.gradient(T_true, z, axis=1)
    return dict(w_true=w_true, w_est=w_est, A_true=A_true, A_est=A_est,
                z=z, T_true=T_true, dTdz_true=dTdz_true)


def load_fields():
    return load_cell(os.path.join(DATA, f"{CONFIG}__cell_{CELL}.nc"))


# ---------------------------------------------------------------- panel helpers
def _psd_contourf(ax, fr, kz, S, vmin, vmax):
    """Filled-contour PSD panel on log-log axes (101 log-spaced levels)."""
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 101)
    pc = ax.contourf(fr, kz, S.T.clip(vmin), levels=levels,
                     norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmo.thermal,
                     extend="both")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(right=FREQ_MAX)
    return pc


def _ratio_contourf(ax, fr, kz, St, Se):
    """log2(estimate/true) variance ratio panel, diverging about 0."""
    ratio = np.log2(np.where(St > 0, Se, np.nan) / np.where(St > 0, St, np.nan))
    rp = ax.contourf(fr, kz, ratio.T.clip(-4, 4), levels=np.linspace(-4, 4, 101),
                     cmap=cmo.balance, norm=TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4),
                     extend="both")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(right=FREQ_MAX)
    return rp


def _decade_ticks(cb, vmin, vmax):
    """Clean power-of-ten ticks on a LogNorm colorbar. Ticks are kept strictly
    INSIDE [vmin, vmax] — out-of-range ticks expand the bar axis past the colored
    solids and leave white gaps at both ends."""
    dec = [10.0 ** e for e in range(int(np.ceil(np.log10(vmin))),
                                    int(np.floor(np.log10(vmax))) + 1)]
    cb.set_ticks(dec)
    cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    cb.ax.minorticks_off()


def _psd_colorbar(fig, axes_list, vmin, vmax, label):
    """Smooth (ScalarMappable) log PSD colorbar. set_aspect('auto') lets the bar
    fill its allocated axes (a fixed aspect leaves white gaps at both ends)."""
    sm = ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmo.thermal)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes_list, location="right", pad=0.01, shrink=0.9)
    cb.ax.set_aspect("auto")
    _decade_ticks(cb, vmin, vmax)
    cb.set_label(label, fontsize=8)
    return cb


def _ratio_colorbar(fig, axes_list):
    sm = ScalarMappable(norm=TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4),
                        cmap=cmo.balance)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes_list, location="right", pad=0.01, shrink=0.9,
                      ticks=[-4, -2, 0, 2, 4])
    cb.ax.set_aspect("auto")
    cb.set_label(r"$\log_2$(est / true)", fontsize=10)
    return cb


def _context_panel(ax, f, title=None):
    """Stratification context: true footprint-mean dT/dz (+-1 sigma_t) and T vs depth."""
    z = f["z"]
    g = f["dTdz_true"]
    gm, gs_ = g.mean(0), g.std(0)
    ax.fill_betweenx(z, gm - gs_, gm + gs_, color="C0", alpha=0.25,
                     label=r"$\pm1\,\sigma_t$")
    l_grad, = ax.plot(gm, z, color="C0", lw=2.2, label=r"mean $\partial_z T$")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$\partial_z T$  ($\mathrm{^\circ C\,m^{-1}}$)", color="C0")
    ax.tick_params(axis="x", labelcolor="C0")
    ax.set_ylabel("depth (m)")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    axT = ax.twiny()
    l_temp, = axT.plot(f["T_true"].mean(0), z, color="C3", lw=2.2,
                       label=r"mean $T$")
    axT.set_xlabel(r"$T$  ($\mathrm{^\circ C}$)", color="C3")
    axT.tick_params(axis="x", labelcolor="C3")
    ax.legend(handles=[l_grad, l_temp], loc="lower right", fontsize=10)


PSD_UNIT_W = r"$(\mathrm{m\,day^{-1}})^2\,\mathrm{cpd^{-1}\,cpm^{-1}}$"
PSD_UNIT_H = r"$(\mathrm{^\circ C\,day^{-1}})^2\,\mathrm{cpd^{-1}\,cpm^{-1}}$"
YLAB_1D_W = r"PSD  $[(\mathrm{m\,day^{-1}})^2\,\mathrm{cpd^{-1}}]$"
YLAB_1D_H = r"PSD  $[(\mathrm{^\circ C\,day^{-1}})^2\,\mathrm{cpd^{-1}}]$"


def _decorate_1d(ax, title, ylab, L, ylim):
    """Shared 1D-panel decoration: TIW band, shared x/y limits, labels."""
    ax.axvspan(1 / 33, 1 / 17, color="0.6", alpha=0.12)         # TIW band
    ax.text(1 / 25, 0.02, "TIW", color="0.4", fontsize=9, ha="center",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(*L["x"]); ax.set_ylim(*ylim)
    ax.set_xlabel("frequency (cpd)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def fig_1d(f, L):
    """Depth-averaged frequency spectra, true vs estimate, both quantities."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kt, ke, title, ylab, ylim in [
        (axes[0], "w_true", "w_est", r"area-mean $w$", YLAB_1D_W, L["y_w"]),
        (axes[1], "A_true", "A_est", r"advective heating  $w\,\partial_z T$",
         YLAB_1D_H, L["y_h"])]:
        fr, pt = psd_1d(f[kt])
        _, pe = psd_1d(f[ke])
        ax.loglog(fr, pt, color="0.15", lw=2.2, label="true")
        ax.loglog(fr, pe, color="C3", lw=1.8, ls="--", label="estimate")
        _decorate_1d(ax, title, ylab, L, ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, f"spectra_1d_{CONFIG}_cell_{CELL}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_2d(f, L):
    """2D vertical-wavenumber-frequency spectra; rows w / heating,
    cols True, Estimate, Estimate/True variance ratio. Colorbars are global."""
    rows = [("w_true", "w_est", r"area-mean $w$", L["vmin_w"], L["vmax_w"], PSD_UNIT_W),
            ("A_true", "A_est", r"$w\,\partial_z T$", L["vmin_h"], L["vmax_h"], PSD_UNIT_H)]
    fig = plt.figure(figsize=(18, 9), layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[0.5, 1, 1, 1])
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(3)] for r in range(2)]
    _context_panel(fig.add_subplot(gs[:, 0]), f)
    for r, (kt, ke, rlab, vmin, vmax, unit) in enumerate(rows):
        fr, kz, St = psd_2d(f[kt])
        _, _, Se = psd_2d(f[ke])
        _psd_contourf(axes[r][0], fr, kz, St, vmin, vmax)
        _psd_contourf(axes[r][1], fr, kz, Se, vmin, vmax)
        _ratio_contourf(axes[r][2], fr, kz, St, Se)
        axes[r][0].set_ylabel("vertical wavenumber (cpm)")
        axes[r][0].set_title(f"{rlab} — True")
        axes[r][1].set_title(f"{rlab} — Estimate")
        axes[r][2].set_title(f"{rlab} — est/true")
        for c in range(3):
            axes[r][c].set_xlabel("frequency (cpd)")
        _psd_colorbar(fig, [axes[r][0], axes[r][1]], vmin, vmax, "PSD  " + unit)
    _ratio_colorbar(fig, [axes[0][2], axes[1][2]])
    out = os.path.join(OUT, f"spectra_2d_{CONFIG}_cell_{CELL}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------ shift-array (per-latitude) figs
def fig_1d_shift(lat, data, L):
    """Depth-averaged frequency spectra at one latitude, 3 shapes overlaid
    (color) x true (solid) / estimate (dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kt, ke, title, ylab, ylim in [
        (axes[0], "w_true", "w_est", r"area-mean $w$", YLAB_1D_W, L["y_w"]),
        (axes[1], "A_true", "A_est", r"advective heating  $w\,\partial_z T$",
         YLAB_1D_H, L["y_h"])]:
        for s in SHAPES:
            fr, pt = psd_1d(data[s][kt])
            _, pe = psd_1d(data[s][ke])
            ax.loglog(fr, pt, color=SHAPE_COLOR[s], lw=2.0)
            ax.loglog(fr, pe, color=SHAPE_COLOR[s], lw=1.6, ls="--")
        _decorate_1d(ax, f"{title}   ({lat:+.1f}" + r"$^\circ$N)", ylab, L, ylim)
    handles = ([Line2D([], [], color=SHAPE_COLOR[s], lw=2.4, label=s) for s in SHAPES]
               + [Line2D([], [], color="0.2", lw=2.2, label="true"),
                  Line2D([], [], color="0.2", lw=1.8, ls="--", label="estimate")])
    fig.legend(handles=handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, f"spectra_1d_shift_{lat:+.1f}deg.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_2d_shift(lat, data, L):
    """2D vertical-wavenumber-frequency spectra at one latitude. Rows = shapes;
    columns = [context | w True | w Est | w.dzT True | w.dzT Est |
    w est/true | w.dzT est/true]. Colorbars are global (shared across all figures)."""
    specs = {}
    for s in SHAPES:
        fr, kz, wT = psd_2d(data[s]["w_true"])
        _, _, wE = psd_2d(data[s]["w_est"])
        _, _, hT = psd_2d(data[s]["A_true"])
        _, _, hE = psd_2d(data[s]["A_est"])
        specs[s] = dict(fr=fr, kz=kz, wT=wT, wE=wE, hT=hT, hE=hE)

    fig = plt.figure(figsize=(22, 10), layout="constrained")
    gs = fig.add_gridspec(3, 7, width_ratios=[0.55, 1, 1, 1, 1, 1, 1])
    _context_panel(fig.add_subplot(gs[:, 0]), data["diamond"],
                   title=f"{lat:+.1f}" + r"$^\circ$N")
    # columns: 0 w-True 1 w-Est 2 wdzT-True 3 wdzT-Est 4 w-ratio 5 wdzT-ratio
    axS = [[fig.add_subplot(gs[r, c + 1]) for c in range(6)] for r in range(3)]
    for r, s in enumerate(SHAPES):
        sp = specs[s]
        _psd_contourf(axS[r][0], sp["fr"], sp["kz"], sp["wT"], L["vmin_w"], L["vmax_w"])
        _psd_contourf(axS[r][1], sp["fr"], sp["kz"], sp["wE"], L["vmin_w"], L["vmax_w"])
        _psd_contourf(axS[r][2], sp["fr"], sp["kz"], sp["hT"], L["vmin_h"], L["vmax_h"])
        _psd_contourf(axS[r][3], sp["fr"], sp["kz"], sp["hE"], L["vmin_h"], L["vmax_h"])
        _ratio_contourf(axS[r][4], sp["fr"], sp["kz"], sp["wT"], sp["wE"])
        _ratio_contourf(axS[r][5], sp["fr"], sp["kz"], sp["hT"], sp["hE"])
        axS[r][0].set_ylabel(f"{s}\n(vert. wavenumber, cpm)",
                             color=SHAPE_COLOR[s], fontweight="bold", fontsize=12)
    titles = [r"area-mean $w$ — True", r"area-mean $w$ — Estimate",
              r"$w\,\partial_z T$ — True", r"$w\,\partial_z T$ — Estimate",
              r"area-mean $w$ — est/true", r"$w\,\partial_z T$ — est/true"]
    for c, t in enumerate(titles):
        axS[0][c].set_title(t, fontsize=11)
    for c in range(6):
        axS[2][c].set_xlabel("frequency (cpd)")
    _psd_colorbar(fig, [axS[r][c] for r in range(3) for c in (0, 1)],
                  L["vmin_w"], L["vmax_w"], r"$w$ PSD  " + PSD_UNIT_W)
    _psd_colorbar(fig, [axS[r][c] for r in range(3) for c in (2, 3)],
                  L["vmin_h"], L["vmax_h"], r"$w\partial_zT$ PSD  " + PSD_UNIT_H)
    _ratio_colorbar(fig, [axS[r][c] for r in range(3) for c in (4, 5)])
    out = os.path.join(OUT, f"spectra_2d_shift_{lat:+.1f}deg.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def compute_global_limits(all_fields):
    """Shared limits so every figure is directly comparable: 1D x/y limits per
    quantity and 2D PSD color ranges, taken over all configs (within f <= FREQ_MAX)."""
    wy, hy = [], []
    xmin = np.inf
    vmax_w = vmax_h = 0.0
    for f in all_fields:
        for k in ("w_true", "w_est"):
            fr, p = psd_1d(f[k]); m = fr <= FREQ_MAX
            wy.append(p[m]); xmin = min(xmin, fr[0])
            _, _, S = psd_2d(f[k]); vmax_w = max(vmax_w, S.max())
        for k in ("A_true", "A_est"):
            fr, p = psd_1d(f[k]); m = fr <= FREQ_MAX
            hy.append(p[m])
            _, _, S = psd_2d(f[k]); vmax_h = max(vmax_h, S.max())

    def ylim(chunks):
        v = np.concatenate(chunks); v = v[v > 0]
        return (v.min() * 0.5, v.max() * 2.0)

    return dict(x=(xmin, FREQ_MAX), y_w=ylim(wy), y_h=ylim(hy),
                vmin_w=vmax_w * 1e-3, vmax_w=vmax_w,
                vmin_h=vmax_h * 1e-3, vmax_h=vmax_h)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    equator = load_fields()
    shift_data = {}
    for lat in SHIFT_LATS:
        paths = {s: shift_path(s, lat) for s in SHAPES}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"skip {lat:+.1f}: missing files")
            continue
        shift_data[lat] = {s: load_cell(p) for s, p in paths.items()}

    all_fields = [equator] + [f for d in shift_data.values() for f in d.values()]
    L = compute_global_limits(all_fields)

    fig_1d(equator, L)
    fig_2d(equator, L)
    for lat, data in shift_data.items():
        fig_1d_shift(lat, data, L)
        fig_2d_shift(lat, data, L)
