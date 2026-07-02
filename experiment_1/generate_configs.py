#!/usr/bin/env python
"""
generate_configs.py — write every experiment_1 array configuration as JSON.

Single source of truth for the OSSE config grid. Placement rules:
  * TAO moorings sit on the 140W line (lon 220) at lat in {-2,-1,0,1,2};
    reusing a mooring is "free" (no glider spent).
  * Gliders are placed at symmetric longitude offsets 220 +/- off.
  * Total glider budget is 6.

Each config file lists the union of `positions` and one or more `cells`
(each a center_lat plus its own point set) — one independent w estimate per cell.
Top-level metadata (family, width, gliders-per-cell, cell height) is copied into
the metrics table by run_experiment.py, so it must stay accurate here.

Families written:
  A  shift      north/south-shifted 3-cell diamond arrays (widths extended to 2.0)
  B  equator    equator-centred estimates: 2deg (mooring-spanned), 1deg (glider-spanned),
                and a symmetric 3-cell array centred at -1/0/+1
  C  density    fixed centre (0.5N), gliders-per-cell swept 2/4/6 at each width

Run:  python generate_configs.py   ->   experiment_1/configs/**/*.json
"""
import json
import os

LON = 220.0                                    # 140W mooring line
WIDTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]      # glider longitude offset (deg)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_ROOT = os.path.join(HERE, 'configs')


def _wkey(off):
    """'w0.25', 'w0.5', ... 'w2.0' — matches the existing config naming."""
    return f"w{off}"


def _n_gliders_total(cells):
    """Unique glider points (lon != mooring line) across all cells of a config."""
    pts = {(round(p[0], 6), round(p[1], 6))
           for c in cells for p in c['positions'] if abs(p[1] - LON) > 1e-9}
    return len(pts)


def _union_positions(cells):
    seen, out = set(), []
    for c in cells:
        for p in c['positions']:
            key = (round(p[0], 6), round(p[1], 6))
            if key not in seen:
                seen.add(key)
                out.append([float(p[0]), float(p[1])])
    return out


def _write(subdir, doc):
    d = os.path.join(CFG_ROOT, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, doc['name'] + '.json')
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)
    return os.path.relpath(path, HERE)


# --- cell builders (each returns a list of {center_lat, positions}) ----------

def _shift_cells(centers, off):
    """1deg-tall diamonds: N/S moorings at center +/-0.5, E/W gliders at +/-off."""
    return [{'center_lat': c,
             'positions': [[c - 0.5, LON], [c + 0.5, LON],
                           [c, LON - off], [c, LON + off]]}
            for c in centers]


def _equator_cell(off, half_height, center_mooring=True):
    """Single equator-centred diamond; N/S at +/-half_height, E/W gliders at +/-off."""
    pos = [[half_height, LON], [-half_height, LON],
           [0.0, LON - off], [0.0, LON + off]]
    if center_mooring:
        pos.append([0.0, LON])            # anchors the fit / checks linearity; interior to hull
    return [{'center_lat': 0.0, 'positions': pos}]


def _equator_3cell(off):
    """Symmetric no-shift array: 2deg-tall diamonds centred at -1/0/+1 (6 gliders)."""
    return [{'center_lat': c,
             'positions': [[c - 1.0, LON], [c + 1.0, LON],
                           [c, LON - off], [c, LON + off]]}
            for c in (-1.0, 0.0, 1.0)]


def _density_cell(off, rows):
    """Fixed centre 0.5N: N/S moorings at 0/1, glider rows at `rows` (each +/-off)."""
    pos = [[0.0, LON], [1.0, LON]]
    for r in rows:
        pos += [[r, LON - off], [r, LON + off]]
    return [{'center_lat': 0.5, 'positions': pos}]


def build():
    written = []
    for off in WIDTHS:
        wk = _wkey(off)

        # A. shift arrays -------------------------------------------------
        for pattern, centers in (('north', [-0.5, 0.5, 1.5]),
                                  ('south', [-1.5, -0.5, 0.5])):
            cells = _shift_cells(centers, off)
            written.append(_write(f'shift/{pattern}', dict(
                name=f'shift_{pattern}_{wk}', family='shift', pattern=pattern,
                width=off, n_gliders_per_cell=2, n_gliders_total=_n_gliders_total(cells),
                cell_height_deg=1.0,
                description=(f'{pattern}-shifted 3-cell diamonds, glider lon offset {off} deg; '
                             f'cells centred at {centers}.'),
                positions=_union_positions(cells), cells=cells)))

        # B. equator-centred ---------------------------------------------
        for tag, half in (('2deg', 1.0), ('1deg', 0.5)):
            cells = _equator_cell(off, half)
            written.append(_write('equator', dict(
                name=f'equator_{tag}_{wk}', family='equator', pattern=f'equator_{tag}',
                width=off, n_gliders_per_cell=_n_gliders_total(cells),
                n_gliders_total=_n_gliders_total(cells), cell_height_deg=2 * half,
                description=(f'Equator-centred diamond, {2 * half:g}deg tall, glider lon '
                             f'offset {off} deg, with centre mooring.'),
                positions=_union_positions(cells), cells=cells)))

        cells = _equator_3cell(off)
        written.append(_write('equator', dict(
            name=f'equator_3cell_{wk}', family='equator', pattern='equator_3cell',
            width=off, n_gliders_per_cell=2, n_gliders_total=_n_gliders_total(cells),
            cell_height_deg=2.0,
            description=(f'Symmetric no-shift 3-cell array, 2deg-tall diamonds centred at '
                         f'-1/0/+1, glider lon offset {off} deg.'),
            positions=_union_positions(cells), cells=cells)))

        # C. density sweep at fixed centre (0.5N) ------------------------
        for ng, rows in ((2, [0.5]), (4, [0.25, 0.75]), (6, [0.25, 0.5, 0.75])):
            cells = _density_cell(off, rows)
            written.append(_write('density', dict(
                name=f'density_{ng}g_{wk}', family='density', pattern=f'density_{ng}g',
                width=off, n_gliders_per_cell=ng, n_gliders_total=ng,
                cell_height_deg=1.0,
                description=(f'Fixed centre 0.5N, {ng} gliders/cell (rows {rows}), '
                             f'glider lon offset {off} deg.'),
                positions=_union_positions(cells), cells=cells)))
    return written


if __name__ == '__main__':
    files = build()
    print(f'Wrote {len(files)} configs under {CFG_ROOT}')
    for f in files:
        print('  ', f)
