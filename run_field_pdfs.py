"""
Worker: field-distribution comparison for one array config at one depth.

Usage: python run_field_pdfs.py <config.json> <max_depth> <tstep>
Writes figures and stats.json to field_pdfs/<max_depth>m/<config name>/.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from osse_tools import (load_model, load_positions, sample_fields, model_region,
                        add_density, eddy_anomalies, compute_w_planefit,
                        model_divergence, dist_stats, plot_field_pdfs,
                        plot_joint_compare, plot_pdf_compare)

RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS = list(range(36, 26173, 36))


def main(cfg_file, max_depth, tstep):
    with open(cfg_file) as f:
        name = json.load(f)['name']
    positions = load_positions(cfg_file)
    outdir = f'field_pdfs/{max_depth}m/{name}'
    os.makedirs(outdir, exist_ok=True)

    ds = load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
    ds = ds.isel(time=slice(None, None, tstep))

    obs = eddy_anomalies(add_density(sample_fields(ds, positions, max_depth=max_depth))).compute()
    true = eddy_anomalies(add_density(model_region(ds, positions, max_depth=max_depth))).compute()

    fig = plot_field_pdfs(obs, true)
    fig.suptitle(f'{name}: field PDFs  (0-{max_depth} m)', y=1.01)
    fig.savefig(f'{outdir}/field_pdfs.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    pairs = [
        (obs.U,  obs.V,  true.U,  true.V,  ('U', 'V'),   ('m/s', 'm/s'), 'uv'),
        (obs.T,  obs.S,  true.T,  true.S,  ('T', 'S'),   ('°C', 'g/kg'), 'TS'),
        (obs.Up, obs.Vp, true.Up, true.Vp, ("u'", "v'"), ('m/s', 'm/s'), 'stress_uv'),
        (obs.Vp, obs.Tp, true.Vp, true.Tp, ("v'", "T'"), ('m/s', '°C'),  'heatflux_vT'),
        (obs.Up, obs.Tp, true.Up, true.Tp, ("u'", "T'"), ('m/s', '°C'),  'heatflux_uT'),
    ]
    for ox, oy, tx, ty, labels, units, tag in pairs:
        fig = plot_joint_compare(ox, oy, tx, ty, labels, units)
        fig.suptitle(f'{name}: {labels[0]}-{labels[1]}', y=1.02)
        fig.savefig(f'{outdir}/joint_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # divergence: plane-fit estimate vs true area mean
    div_obs = compute_w_planefit(
        sample_fields(ds, positions, vars=('UVEL', 'VVEL'), max_depth=max_depth))['div']
    div_true = model_divergence(ds, positions, max_depth=max_depth)
    fig = plot_pdf_compare(div_obs, div_true, label='divergence', units='1/s')
    fig.suptitle(f'{name}: horizontal divergence', y=1.02)
    fig.savefig(f'{outdir}/divergence_pdf.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    div_corr = float(np.corrcoef(div_obs.values.ravel(), div_true.values.ravel())[0, 1])

    stats = {'name': name, 'max_depth': max_depth, 'tstep': tstep,
             'n_glider': int(obs.sizes['glider']), 'n_point': int(true.sizes['point']),
             'n_time': int(obs.sizes['time']), 'div_corr': div_corr, 'fields': {}}
    for v in ('T', 'S', 'U', 'V', 'sigma0'):
        stats['fields'][v] = dist_stats(obs[v], true[v])
    stats['fields']['divergence'] = dist_stats(div_obs, div_true)
    with open(f'{outdir}/stats.json', 'w') as f:
        json.dump(stats, f, indent=1, default=float)

    sys.stderr.write(f"DONE {name} {max_depth}m  div_corr={div_corr:.2f}\n")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
