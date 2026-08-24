"""
Compute the OCV flux budget and cache it, reading the dt60 model ONCE.

For every symhex diameter (centre 0degN,140degW) this writes cache/<name>_ocv.nc holding
the space-averaged series the study compares as *temporal* PDFs:

  Per-face outward-normal current & advective heat flux  (time, face, obs_depth):
    un_true / un_obs      outward-normal current  [m/s]
    unT_true / unT_obs    outward advective heat flux  [degC m/s]  (x HFLUX -> W/m^2)
  truth = edge-average over every high-res model point on the face;
  obs   = single glider at the face centre (midpoint rule).

  OCV volume(area)-mean fields  (time, obs_depth):
    Tbar_true/Sbar_true/Ubar_true/Vbar_true/Wbar_true   interior mean over the polygon
    Tbar_obs/Sbar_obs/Ubar_obs/Vbar_obs                 mean of face gliders + mooring
  Wbar_true is the model's directly-diagnosed area-mean w -- the validation overlay.

  Vertically-integrated budgets  (time, depth interfaces):
    w_true / w_obs        area-averaged w from the divergence theorem (continuity)
    wT_true / wT_obs       vertical advective heat flux from the heat budget with storage
                           [W/m^2]

See README.md for the divergence-theorem / midpoint-rule derivation.
"""

import os
import numpy as np
import xarray as xr

import common as C
import osse_tools as ot

DT_SEC = C.ITER_STEP * C.DELTA_T          # 3-hourly output spacing (s) for storage dT/dt
_KEEP = {'time', 'face', 'obs_depth', 'depth'}


def _clean(da):
    """Drop the stray MITgcm grid coords (drF, rA, ...) that interp leaves behind, so
    variables from different staggered grids merge without coordinate conflicts."""
    return da.drop_vars([c for c in da.coords if c not in _KEEP])


# ------------------------------------------------------------------ budget integration
def _integrate(series, dz):
    """Integrate a midpoint (time, obs_depth) series down the column -> w at interfaces.

    w(interface k) = cumsum over layers above k of (series * dz), with w=0 at the top
    (surface once the inputs are extrapolated there). Mirrors compute_w_planefit."""
    s = series.transpose('time', 'obs_depth')
    obs_z = s.obs_depth.values
    n = obs_z.size
    z_top = obs_z[0] + dz / 2.0
    w_z = z_top - np.arange(n + 1) * dz
    vals = s.values
    w = np.concatenate([np.zeros((vals.shape[0], 1)),
                        np.cumsum(vals * dz, axis=1)], axis=1)
    return xr.DataArray(w, dims=('time', 'depth'),
                        coords={'time': s.time, 'depth': w_z})


def _budget(un, unT, Tbar, area, L_da):
    """From per-face outward series + OCV-mean T, build the continuity and heat budgets.

    un, unT : (time, face, obs_depth) outward-normal current / heat flux.
    Tbar    : (time, obs_depth) OCV volume-mean temperature (for the storage term).
    Returns w (m/s) and wT (W/m^2) at interfaces, plus the divergence diagnostics.
    """
    fn = xr.Dataset({'un': _clean(un), 'unT': _clean(unT), 'Tbar': _clean(Tbar)})
    fn = ot.extrapolate_currents_to_surface(fn)          # fill 0-8 m by top shear -> w=0 at 0 m
    dz = C.DZ_OBS
    div_area = (fn.un * L_da).sum('face') / area          # (1/A) sum un*L  = area-mean divergence [1/s]
    hflux_div = (fn.unT * L_da).sum('face') / area        # (1/A) sum unT*L                       [degC/s]
    tax = fn.Tbar.get_axis_num('time')
    dTdt = fn.Tbar.copy(data=np.gradient(fn.Tbar.values, DT_SEC, axis=tax))  # storage [degC/s]
    w = _integrate(div_area, dz)
    wT = _integrate(dTdt + hflux_div, dz) * C.HFLUX       # degC m/s -> W/m^2
    return w, wT, div_area, hflux_div


# ------------------------------------------------------------------ per-config compute
def compute_config(ds, region, diam, shape='symhex', grid_deg=None):
    name = C.config_name(diam, shape)
    fg = C.face_geometry(diam, shape)
    A, L = fg.area, fg.lengths
    nx = xr.DataArray(fg.normals[:, 0], dims='face')
    ny = xr.DataArray(fg.normals[:, 1], dims='face')
    L_da = xr.DataArray(L, dims='face')
    face_lat = xr.DataArray([c[0] for c in fg.face_centers], dims='face')
    face_lon = xr.DataArray([c[1] for c in fg.face_centers], dims='face')

    # --- TRUTH face-averages: dense edge sampling, mean over each edge --------------
    epos, nf, npf = C.edge_sample_positions(diam, shape, grid_deg=grid_deg)
    se = C.sample_points(ds, epos).compute()             # (time, glider=nf*npf, obs_depth)
    ntime, _, nobs = se.U.shape
    Ue = se.U.values.reshape(ntime, nf, npf, nobs)
    Ve = se.V.values.reshape(ntime, nf, npf, nobs)
    Te = se.T.values.reshape(ntime, nf, npf, nobs)
    nxf = fg.normals[:, 0][None, :, None, None]
    nyf = fg.normals[:, 1][None, :, None, None]
    un_pt = Ue * nxf + Ve * nyf                          # (time, face, edge_pt, obs)
    coords = {'time': se.time, 'face': np.arange(nf), 'obs_depth': se.obs_depth}
    un_true = xr.DataArray(un_pt.mean(axis=2), dims=('time', 'face', 'obs_depth'),
                           coords=coords)
    unT_true = xr.DataArray((un_pt * Te).mean(axis=2), dims=('time', 'face', 'obs_depth'),
                            coords=coords)

    # --- OBS: face-centre gliders (+ mooring for the volume mean) --------------------
    sa = C.sample_points(ds, C.array_positions(diam, shape)).compute()  # last glider = mooring
    mi = C.mooring_index(diam, shape)
    g = sa.rename({'glider': 'face'}).isel(face=slice(0, mi))
    un_obs = (g.U * nx + g.V * ny).transpose('time', 'face', 'obs_depth')
    unT_obs = (un_obs * g.T).transpose('time', 'face', 'obs_depth')

    # --- OCV volume(area) means -----------------------------------------------------
    sel = C.select_polygon(region, fg.vertices)
    true_mean = {v + 'bar_true': sel[v].mean('point') for v in ('T', 'S', 'U', 'V', 'W')}
    obs_mean = {v + 'bar_obs': sa[v].mean('glider') for v in ('T', 'S', 'U', 'V')}

    # --- budgets --------------------------------------------------------------------
    w_true, wT_true, div_true, hdiv_true = _budget(un_true, unT_true,
                                                    true_mean['Tbar_true'], A, L_da)
    w_obs, wT_obs, div_obs, hdiv_obs = _budget(un_obs, unT_obs,
                                               obs_mean['Tbar_obs'], A, L_da)

    data = dict(
        un_true=un_true, un_obs=un_obs, unT_true=unT_true, unT_obs=unT_obs,
        w_true=w_true, w_obs=w_obs, wT_true=wT_true, wT_obs=wT_obs,
        div_true=div_true, div_obs=div_obs, hdiv_true=hdiv_true, hdiv_obs=hdiv_obs,
        **true_mean, **obs_mean,
    )
    out = xr.Dataset({k: _clean(v) for k, v in data.items()})
    out = out.assign_coords(face_lat=face_lat, face_lon=face_lon,
                            face_len=('face', L))
    out.attrs.update(config=name, diameter=diam, shape=shape, area=A,
                     n_faces=nf, n_edge_pts=npf, mooring_index=mi)
    out.to_netcdf(os.path.join(C.CACHE_DIR, f'{name}_ocv.nc'))

    # --- sanity: truth-integrated w should track the model's area-mean w -----------
    z0 = -70.0
    wt = C.SEC_PER_DAY * w_true.interp(depth=z0).mean('time').item()
    wm = C.SEC_PER_DAY * true_mean['Wbar_true'].interp(obs_depth=z0).mean('time').item()
    r = float(np.corrcoef(w_true.interp(depth=z0).values,
                          true_mean['Wbar_true'].interp(obs_depth=z0).values)[0, 1])
    print(f'  {name}: A={A:.3e} m^2, faces={nf}, edge_pts={npf} | '
          f'w@{-z0:.0f}m truth-int={wt:+.3f} model={wm:+.3f} m/day, r={r:.3f}')
    return out


def main(diams=None, shapes=None):
    diams = diams or C.DIAMETERS
    shapes = shapes or C.SHAPES
    # bbox must reach the OUTERMOST vertex: with the circle inscribed, vertices sit at
    # circumradius (D/2)/cos(pi/n) -- e.g. the diamond's tips reach 0.707 deg at D=1, well
    # beyond max(D)/2. Size the half-width from the actual geometry so no edge point is
    # sampled outside the loaded region.
    half = max(np.hypot(v[0] - C.CENTER[0], v[1] - C.CENTER[1])
               for shape in shapes for d in diams for v in C.ocv_vertices(d, shape))
    print(f'loading bbox (half={half:.3f} deg) into memory ...')
    ds = C.load_bbox_memory(half)
    grid_deg = C.native_grid_deg(ds)
    print(f'native grid spacing ~{grid_deg:.4f} deg; co-locating region ...')
    region = C.region_bbox(ds, half).compute()
    for shape in shapes:
        for d in diams:
            compute_config(ds, region, d, shape, grid_deg=grid_deg)
    print('done ->', C.CACHE_DIR)


if __name__ == '__main__':
    main()
