"""
tests for osse_tools.py.

Run with: conda run -n tpose python test_osse_tools.py
"""

import numpy as np
import xarray as xr
import sys


def _hexagon_positions(lat_c, lon_c, radius_deg):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(lat_c + radius_deg * np.sin(a), lon_c + radius_deg * np.cos(a))
            for a in angles]


def _fake_uv_samples(positions, div_target, ntime=3, nz=10, dz=1.0):
    """
    Build a uv_samples Dataset with a linear velocity field producing div_target.
    u = du_dx * x,  v = dv_dy * y  so that du_dx + dv_dy = div_target.
    Split evenly: du_dx = dv_dy = div_target / 2.
    """
    from osse_tools import compute_w_planefit  # noqa — import inside to keep test isolated

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
    Z = -np.arange(nz) - 0.5  # cell centers

    # u_i = du_dx * x_i, broadcast over time and Z
    U_vals = (du_dx * x_m)[np.newaxis, :, np.newaxis] * np.ones((ntime, n, nz))
    V_vals = (dv_dy * y_m)[np.newaxis, :, np.newaxis] * np.ones((ntime, n, nz))

    lat_da = xr.DataArray(lats, dims='glider', coords={'glider': g})
    lon_da = xr.DataArray(lons, dims='glider', coords={'glider': g})

    U = xr.DataArray(U_vals, dims=('time', 'glider', 'Z'),
                     coords={'time': time, 'glider': g, 'Z': Z})
    V = xr.DataArray(V_vals, dims=('time', 'glider', 'Z'),
                     coords={'time': time, 'glider': g, 'Z': Z})

    uv = xr.Dataset({'U': U, 'V': V}).assign_coords(lat=lat_da, lon=lon_da)

    Zl = -np.arange(nz)
    fake_ds = xr.Dataset({
        'drF': xr.DataArray(np.full(nz, dz), dims='Z', coords={'Z': Z}),
        'Zl':  xr.DataArray(Zl, dims='Zl'),
    })

    return uv, fake_ds


def test_planefit_div_recovery():
    """Plane fit should exactly recover divergence from a linear velocity field."""
    from osse_tools import compute_w_planefit

    positions = _hexagon_positions(0.0, 220.0, 0.125)
    div_target = 1e-5  # 1/s

    uv, fake_ds = _fake_uv_samples(positions, div_target)
    result = compute_w_planefit(uv, fake_ds)

    div_recovered = result['div'].values  # (ntime, nz)
    assert np.allclose(div_recovered, div_target, rtol=1e-6), (
        f"div mismatch: expected {div_target:.2e}, got {div_recovered.mean():.2e}"
    )
    print(f"  div recovery: target={div_target:.2e}, recovered={div_recovered.mean():.2e}  OK")


def test_planefit_w_integration():
    """Integrated w should equal -div * cumulative depth."""
    from osse_tools import compute_w_planefit

    positions = _hexagon_positions(0.0, 220.0, 0.125)
    div_target = 1e-5
    dz = 1.0
    nz = 10

    uv, fake_ds = _fake_uv_samples(positions, div_target, nz=nz, dz=dz)
    result = compute_w_planefit(uv, fake_ds)

    w = result['w_est'].values  # (ntime, nz)
    # Expected: w at Zl[k] = -div * k * dz
    expected = -div_target * np.arange(nz) * dz
    assert np.allclose(w[0], expected, rtol=1e-6), (
        f"w integration error:\n  expected={expected[:5]}\n  got={w[0, :5]}"
    )
    print(f"  w integration: w at Zl[5]={w[0,5]:.2e}, expected={expected[5]:.2e}  OK")


def test_sample_uv_shape():
    """sample_uv output should have the right dimensions and coordinates."""
    # This test uses a tiny slice of real model data (1 iter, small z subset)
    from osse_tools import load_model, sample_uv

    run_dir = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
    ds = load_model(run_dir, iters=[36])
    positions = _hexagon_positions(0.0, 220.0, 0.125)
    uv = sample_uv(ds, positions)

    assert uv['U'].dims == ('time', 'glider', 'Z'), f"unexpected dims: {uv['U'].dims}"
    assert uv['U'].shape[1] == 6, f"expected 6 gliders, got {uv['U'].shape[1]}"
    assert 'lat' in uv.coords and 'lon' in uv.coords
    print(f"  sample_uv shape: {uv['U'].shape}  OK")


def test_sample_model_w_shape():
    """sample_model_w should return (time, Zl)."""
    from osse_tools import load_model, sample_model_w

    run_dir = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
    ds = load_model(run_dir, iters=[36])
    positions = _hexagon_positions(0.0, 220.0, 0.125)
    w_model = sample_model_w(ds, positions)

    assert 'time' in w_model.dims and 'Zl' in w_model.dims, (
        f"unexpected dims: {w_model.dims}"
    )
    print(f"  sample_model_w shape: {w_model.shape}  OK")


if __name__ == '__main__':
    tests = [
        test_planefit_div_recovery,
        test_planefit_w_integration,
        test_sample_uv_shape,
        test_sample_model_w_shape,
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
