"""
Tests for osse_tools.py.

Run with: conda run -n tpose python test_osse_tools.py
"""

import numpy as np
import xarray as xr
import sys


def _hexagon_positions(lat_c, lon_c, radius_deg):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(lat_c + radius_deg * np.sin(a), lon_c + radius_deg * np.cos(a))
            for a in angles]


def _fake_uv_samples(positions, div_target, ntime=3, n_obs=10, dz_obs=2.0):
    """
    Build a uv_samples Dataset with a linear velocity field producing div_target.
    u = du_dx * x,  v = dv_dy * y,  du_dx = dv_dy = div_target / 2.
    """
    lats = np.array([p[0] for p in positions])
    lons = np.array([p[1] for p in positions])
    lat_c, lon_c = lats.mean(), lons.mean()

    deg_to_m = np.pi / 180 * 6371000.0
    x_m = (lons - lon_c) * np.cos(np.radians(lat_c)) * deg_to_m
    y_m = (lats - lat_c) * deg_to_m

    du_dx = div_target / 2
    dv_dy = div_target / 2

    n = len(positions)
    g = np.arange(n)
    time = xr.cftime_range('2012-10-01', periods=ntime, freq='3h')
    obs_z = -(np.arange(n_obs) * dz_obs + dz_obs / 2)  # midpoints

    U_vals = (du_dx * x_m)[np.newaxis, :, np.newaxis] * np.ones((ntime, n, n_obs))
    V_vals = (dv_dy * y_m)[np.newaxis, :, np.newaxis] * np.ones((ntime, n, n_obs))

    lat_da = xr.DataArray(lats, dims='glider', coords={'glider': g})
    lon_da = xr.DataArray(lons, dims='glider', coords={'glider': g})

    U = xr.DataArray(U_vals, dims=('time', 'glider', 'obs_depth'),
                     coords={'time': time, 'glider': g, 'obs_depth': obs_z})
    V = xr.DataArray(V_vals, dims=('time', 'glider', 'obs_depth'),
                     coords={'time': time, 'glider': g, 'obs_depth': obs_z})

    return xr.Dataset({'U': U, 'V': V}).assign_coords(lat=lat_da, lon=lon_da)


def test_planefit_div_recovery():
    """Plane fit should exactly recover divergence from a linear velocity field."""
    from osse_tools import compute_w_planefit

    positions = _hexagon_positions(0.0, 220.0, 0.125)
    div_target = 1e-5

    uv = _fake_uv_samples(positions, div_target)
    result = compute_w_planefit(uv)

    div_recovered = result['div'].values
    assert np.allclose(div_recovered, div_target, rtol=1e-6), (
        f"div mismatch: expected {div_target:.2e}, got {div_recovered.mean():.2e}"
    )
    print(f"  div recovery: target={div_target:.2e}, recovered={div_recovered.mean():.2e}  OK")


def test_planefit_w_integration():
    """Integrated w should equal +div * cumulative depth (positive div → upwelling)."""
    from osse_tools import compute_w_planefit

    positions = _hexagon_positions(0.0, 220.0, 0.125)
    div_target = 1e-5
    dz_obs = 2.0
    n_obs = 10

    uv = _fake_uv_samples(positions, div_target, n_obs=n_obs, dz_obs=dz_obs)
    result = compute_w_planefit(uv)

    w = result['w_est'].values   # (ntime, n_obs+1) at interfaces
    expected = div_target * np.arange(n_obs + 1) * dz_obs
    assert np.allclose(w[0], expected, rtol=1e-6), (
        f"w integration error:\n  expected={expected[:5]}\n  got={w[0, :5]}"
    )
    print(f"  w integration: w at interface 5={w[0,5]:.2e}, expected={expected[5]:.2e}  OK")


def test_sample_fields_shape():
    """sample_fields output should have dims (time, glider, obs_depth) with U,V,T,S."""
    from osse_tools import load_model, sample_fields

    run_dir = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
    ds = load_model(run_dir, iters=[36])
    positions = _hexagon_positions(0.0, 220.0, 0.125)
    uv = sample_fields(ds, positions, max_depth=70, dz_obs=2)

    assert uv['U'].dims == ('time', 'glider', 'obs_depth'), f"unexpected dims: {uv['U'].dims}"
    assert set(uv.data_vars) == {'U', 'V', 'T', 'S'}, f"unexpected vars: {set(uv.data_vars)}"
    assert uv['U'].shape[1] == 6,  f"expected 6 gliders, got {uv['U'].shape[1]}"
    assert uv['U'].shape[2] == 35, f"expected 35 obs levels (70/2), got {uv['U'].shape[2]}"
    assert 'lat' in uv.coords and 'lon' in uv.coords
    print(f"  sample_fields shape: {uv['U'].shape}, vars={set(uv.data_vars)}  OK")


def test_sample_model_w_shape():
    """sample_model_w should return (time, depth) at interface depths."""
    from osse_tools import load_model, sample_model_w

    run_dir = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
    ds = load_model(run_dir, iters=[36])
    positions = _hexagon_positions(0.0, 220.0, 0.125)
    w_model = sample_model_w(ds, positions, max_depth=70, dz_obs=2)

    assert w_model.dims == ('time', 'depth'), f"unexpected dims: {w_model.dims}"
    assert w_model.shape[1] == 36, f"expected 36 interface levels (70/2 + 1), got {w_model.shape[1]}"
    print(f"  sample_model_w shape: {w_model.shape}  OK")


def test_model_region_density_anomalies():
    """model_region truth population, density, and anomalies build correctly."""
    from osse_tools import load_model, model_region, add_density, eddy_anomalies

    run_dir = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
    ds = load_model(run_dir, iters=[36])
    positions = _hexagon_positions(0.0, 220.0, 0.125)
    reg = model_region(ds, positions, max_depth=20, dz_obs=2)

    assert reg['T'].dims == ('time', 'point', 'obs_depth'), f"unexpected dims: {reg['T'].dims}"
    assert 'lat' in reg.coords and 'lon' in reg.coords
    reg = eddy_anomalies(add_density(reg))
    assert 'sigma0' in reg and 'Tp' in reg and 'Vp' in reg
    sig = reg['sigma0'].values
    assert np.nanmin(sig) > 15 and np.nanmax(sig) < 30, f"sigma0 out of range: {np.nanmin(sig)}-{np.nanmax(sig)}"
    print(f"  model_region {reg['T'].shape}, sigma0 in "
          f"[{np.nanmin(sig):.1f}, {np.nanmax(sig):.1f}]  OK")


if __name__ == '__main__':
    tests = [
        test_planefit_div_recovery,
        test_planefit_w_integration,
        test_sample_fields_shape,
        test_sample_model_w_shape,
        test_model_region_density_anomalies,
    ]
    failed = 0
    for t in tests:
        print(f"{t.__name__}")
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed")
