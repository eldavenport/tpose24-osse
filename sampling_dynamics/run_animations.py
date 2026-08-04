"""
Animations: watch the "river of data" flow past the virtual mooring.

  footprint_<diam>.mp4  the model field inside the hexagon footprint at 30 m animated
                        over the record (hull grid points coloured by w'), with the 6
                        glider + mooring sample points overlaid -- the eddies / TIWs /
                        Kelvin waves advecting through the fixed array.
  profiles_<diam>.mp4   vertical profiles of w and kappa_T evolving in time, array
                        estimate (solid) vs model truth (dotted), with a running date.

Both read only the cache from run_sample.py.  mp4 via ffmpeg (falls back to gif).

Usage:  python run_animations.py [diam]   (default 1.0)
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmocean.cm as cmo

import common as C
import sd_plot as P
import run_transport as T

SUB = 'animations'
DAY = C.SEC_PER_DAY


def _save(anim, path_noext, fps=6):        # 6 fps = 50% slower than the original 12
    try:
        anim.save(f'{path_noext}.mp4', writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        return f'{path_noext}.mp4'
    except Exception as e:                                   # pragma: no cover
        sys.stderr.write(f'ffmpeg failed ({e}); writing gif\n')
        anim.save(f'{path_noext}.gif', writer=animation.PillowWriter(fps=fps))
        return f'{path_noext}.gif'


# --------------------------------------------------------------------------- footprint
def anim_footprint(out, diam=1.0, z=-30.0, stride=1):
    cloud = P.load_cloud(diam)
    lon = cloud.lon.values; lat = cloud.lat.values
    w = cloud['W'].interp(obs_depth=z).transpose('time', 'point').values * DAY
    times = cloud.time.values
    days = (times - times[0]) / np.timedelta64(1, 'D')
    gl = C.glider_positions(diam)
    vmax = 25.0                      # fixed w colour range (m day^-1)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(lon, lat, c=w[0], cmap=cmo.balance, vmin=-vmax, vmax=vmax,
                    s=26, marker='s')
    ax.scatter([p[1] for p in gl], [p[0] for p in gl], s=140, facecolors='none',
               edgecolors='k', linewidths=2, label='gliders', zorder=5)
    ax.scatter([C.MOORING[1]], [C.MOORING[0]], s=180, marker='*', color='k',
               label='mooring', zorder=6)
    ax.set(xlabel='longitude ($^\\circ$E)', ylabel='latitude ($^\\circ$N)')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, label="$w$ at 30 m (m day$^{-1}$)")
    ttl = ax.set_title('')

    frames = range(0, len(days), stride)

    def update(i):
        sc.set_array(w[i])
        ttl.set_text(f'{diam:g}$^\\circ$ footprint   day {days[i]:.1f}')
        return sc, ttl

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = _save(anim, f'{out}/footprint_d{diam:g}')
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- profiles
def anim_profiles(out, diam=1.0, stride=2):
    arr = P.load_array(diam); hull = P.load_hull(diam)
    z = arr.obs_depth.values
    w_e = arr['w_est_mid'].values * DAY
    w_t = hull['W'].values * DAY
    k_e = P.array_profile(arr, 'kappaT').values
    k_t = hull['kappaT'].values
    times = arr.time.values
    days = (times - times[0]) / np.timedelta64(1, 'D')

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6))
    wlim = np.nanpercentile(np.abs(np.concatenate([w_e, w_t])), 99)
    klim = np.nanpercentile(np.concatenate([k_e, k_t]), 99.5)
    lw_e, = axes[0].plot(w_e[0], z, color='#08306b', **P.ARRAY_KW)
    lw_t, = axes[0].plot(w_t[0], z, color='#08306b', **P.TRUTH_KW)
    lk_e, = axes[1].plot(k_e[0], z, color='#7a0177', **P.ARRAY_KW)
    lk_t, = axes[1].plot(k_t[0], z, color='#7a0177', **P.TRUTH_KW)
    axes[0].set(xlim=(-wlim, wlim), xlabel='$w$ (m day$^{-1}$)', ylabel='depth (m)')
    axes[1].set(xlim=(1e-5, klim), xscale='log', xlabel=r'$\kappa_T$ (m$^2$ s$^{-1}$)')
    for ax in axes:
        ax.axvline(0 if ax is axes[0] else 1e-5, color='0.7', lw=0.6)
    P.top_legend(fig, diam=False)
    sup = fig.suptitle('')

    frames = range(0, len(days), stride)

    def update(i):
        lw_e.set_xdata(w_e[i]); lw_t.set_xdata(w_t[i])
        lk_e.set_xdata(k_e[i]); lk_t.set_xdata(k_t[i])
        sup.set_text(f'{diam:g}$^\\circ$ footprint   day {days[i]:.1f}')
        return lw_e, lw_t, lk_e, lk_t, sup

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = _save(anim, f'{out}/profiles_d{diam:g}')
    plt.close(fig)
    return path


def main(diam=1.0):
    out = P.outdir(SUB)
    print(anim_footprint(out, 1.0))
    print(anim_footprint(out, 0.3))          # small-footprint companion
    print(anim_profiles(out, diam))


if __name__ == '__main__':
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(d)
