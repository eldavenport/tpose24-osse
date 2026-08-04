"""
Wavenumber-frequency spectra of the virtual mooring's transport quantities.

The footprint average collapses horizontal space, so each quantity is a (time, depth)
series: the "wavenumber" axis is the VERTICAL wavenumber m (cpm) and the other axis is
frequency (cpd).  Data are 3-hourly (frequency Nyquist 4 cpd) on a uniform 2 m grid
(vertical Nyquist 0.25 cpm) over ~80 days.  We spectrally compare, true vs array:
  * area-mean w        truth = hull-mean W,  est = plane-fit w
  * vertical heat flux w'T'   truth = hull-mean w'T',  est = plane-fit w' x array-mean T'

Figures -> spectra/
  spectra_1d.png        depth-averaged frequency PSD of w and w'T', diameters overlaid,
                        true (solid) vs array (dashed); diurnal + TIW band marked.
  spectra_2d_d{d}.png   vertical-wavenumber x frequency PSD, rows [w, w'T'],
                        cols [truth, array, log2 array/truth].

Usage:  python run_spectra.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import cmocean.cm as cmo

import common as C
import sd_plot as P
import run_transport as T

SUB = 'spectra'
DT_DAY = 0.125            # 3-hourly
DZ = 2.0
TIW_BAND = (1 / 40., 1 / 15.)     # ~15-40 day tropical instability waves


def _prep(a2d):
    """Demean per depth, linear-detrend in time, Hann taper in time. a2d = (time, depth)."""
    a = np.asarray(a2d, float)
    a = a - np.nanmean(a, axis=0, keepdims=True)
    t = np.arange(a.shape[0])
    for k in range(a.shape[1]):
        col = a[:, k]
        good = np.isfinite(col)
        if good.sum() > 2:
            p = np.polyfit(t[good], col[good], 1)
            a[:, k] = col - np.polyval(p, t)
    a = np.nan_to_num(a)
    win = np.hanning(a.shape[0])[:, None]
    return a * win, (win ** 2).mean()


def psd_1d(a2d, dt=DT_DAY):
    a, wcorr = _prep(a2d)
    n = a.shape[0]
    F = np.fft.rfft(a, axis=0)
    freq = np.fft.rfftfreq(n, dt)
    psd = (np.abs(F) ** 2).mean(axis=1) * (dt / n / wcorr)    # depth-avg one-sided PSD
    psd[1:-1] *= 2
    return freq[1:], psd[1:]


def psd_2d(a2d, dt=DT_DAY, dz=DZ):
    a, wcorr = _prep(a2d)
    nt, nz = a.shape
    F = np.fft.fftshift(np.fft.fft2(a)) / (nt * nz)
    P2 = np.abs(F) ** 2 / wcorr
    freq = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    kz = np.fft.fftshift(np.fft.fftfreq(nz, dz))
    # fold to positive frequency, sum symmetric kz partners (Parseval-preserving)
    fpos = freq >= 0
    P2 = P2[fpos, :]
    freq = freq[fpos]
    return freq, kz, P2


def _series(diam):
    """(w_true, w_est, wT_true, wT_est) each (time, depth) numpy over 8-80 m."""
    arr = P.load_array(diam); hull = P.load_hull(diam)
    w_est = arr['w_est_mid'].transpose('time', 'obs_depth').values * C.SEC_PER_DAY
    w_true = hull['W'].transpose('time', 'obs_depth').values * C.SEC_PER_DAY
    wT_est = (T.array_vert_flux(arr, 'w_est_mid', 'T').transpose('time', 'obs_depth').values
              * C.HFLUX)
    wT_true = hull['wT'].transpose('time', 'obs_depth').values * C.HFLUX
    return w_true, w_est, wT_true, wT_est


# --------------------------------------------------------------------------- fig 1
def fig_1d(out):
    """Frequency PSD of area-mean w and w'T', ONE ROW PER DIAMETER (true solid vs
    array dashed). Diurnal frequency marked; no band shading."""
    cols = ['area-mean $w$  (m day$^{-1}$)', "vertical heat flux $w'T'$  (W m$^{-2}$)"]
    nr = len(C.DIAMETERS)
    fig, axes = plt.subplots(nr, 2, figsize=(11, 2.7 * nr), sharex=True, sharey='col')
    fmin = None
    for ri, d in enumerate(C.DIAMETERS):
        w_true, w_est, wT_true, wT_est = _series(d)
        c = C.diam_color(d)
        for ci, (true, est) in enumerate([(w_true, w_est), (wT_true, wT_est)]):
            ax = axes[ri, ci]
            f, pt = psd_1d(true); _, pe = psd_1d(est)
            fmin = f.min()
            ax.loglog(f, pt, color=c, lw=2.4)                   # truth = solid
            ax.loglog(f, pe, color=c, lw=1.8, ls='--')          # array = dashed
            ax.axvline(1.0, color='0.5', ls=':', lw=1)          # diurnal
            ax.set_xlim(fmin, 4)
            if ri == 0:
                ax.set_title(cols[ci])
            if ri == nr - 1:
                ax.set_xlabel('frequency (cpd)')
        axes[ri, 0].set_ylabel(f'{d:g}$^\\circ$\nPSD')
    handles = [Line2D([0], [0], color='0.3', lw=2.4, label='truth'),
               Line2D([0], [0], color='0.3', lw=1.8, ls='--', label='array'),
               Line2D([0], [0], color='0.5', ls=':', lw=1, label='diurnal (1 cpd)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    return P.finish(fig, f'{out}/spectra_1d.png')


# --------------------------------------------------------------------------- fig 2
def fig_2d(out, diam=1.0):
    w_true, w_est, wT_true, wT_est = _series(diam)
    rows = [('area-mean $w$', w_true, w_est), ("heat flux $w'T'$", wT_true, wT_est)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for r, (label, true, est) in enumerate(rows):
        f, kz, Pt = psd_2d(true); _, _, Pe = psd_2d(est)
        vmax = np.nanpercentile(Pt, 99.8); vmin = vmax * 1e-3
        norm = LogNorm(vmin=vmin, vmax=vmax)
        kpos = kz >= 0
        for c, (Pdat, ttl, cmap, nn) in enumerate([
                (Pt, 'truth', cmo.thermal, norm), (Pe, 'array', cmo.thermal, norm),
                (np.log2((Pe + vmin) / (Pt + vmin)), 'log$_2$ array/truth', cmo.balance,
                 None)]):
            ax = axes[r, c]
            if nn is None:
                pc = ax.pcolormesh(f, kz[kpos], Pdat[:, kpos].T, cmap=cmap,
                                   vmin=-4, vmax=4, shading='auto')
            else:
                pc = ax.pcolormesh(f, kz[kpos], Pdat[:, kpos].T, cmap=cmap, norm=nn,
                                   shading='auto')
            ax.set_xscale('log'); ax.set_xlim(f[f > 0].min(), 4)
            ax.axvline(1.0, color='w', ls=':', lw=1)
            if r == 1:
                ax.set_xlabel('frequency (cpd)')
            if c == 0:
                ax.set_ylabel(f'{label}\nvert. wavenumber (cpm)')
            ax.set_title(ttl)
            fig.colorbar(pc, ax=ax, pad=0.01)
    return P.finish(fig, f'{out}/spectra_2d_d{diam:g}.png')


def main():
    out = P.outdir(SUB)
    print(fig_1d(out))
    print(fig_2d(out, 1.0))


if __name__ == '__main__':
    main()
