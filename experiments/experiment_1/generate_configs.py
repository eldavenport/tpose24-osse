#!/usr/bin/env python
"""
generate_configs.py — write every experiment_1 array configuration as JSON.

Single source of truth for the OSSE config grid. Placement rules:
  * TAO moorings sit on the 140W line (lon 220) at lat in {-2,-1,0,1,2};
    reusing a mooring is "free" (no glider spent).
  * Gliders are placed at symmetric longitude offsets 220 +/- off.
  * Any real mooring that falls in a cell's interior is added as a sample point
    (it improves the plane fit at no glider cost) — see _interior_moorings.
  * Glider budgets are not strictly capped: some arrays (e.g. the 4-cell shift
    array) spend more than 6 gliders. They are kept because comparing their error
    estimates is informative even if not simultaneously deployable.

Each config file lists the union of `positions` and one or more `cells`
(each a center_lat plus its own point set) — one independent w estimate per cell.
Top-level metadata (family, width, gliders-per-cell, cell height) is copied into
the metrics table by run_experiment.py, so it must stay accurate here.

Families written:
  A  shift      one 4-cell array of 1deg diamonds centred at [-1.5,-0.5,0.5,1.5],
                plus the same 4-cell framework as 1deg hexagons (pattern shift_hex)
  B  equator    equator-centred estimates:
                  - single diamond, 2deg (mooring-spanned) & 1deg (glider-spanned)
                  - single hexagon, 2deg (4 gliders) & 1deg (6 gliders)
                  - single square/box, 2deg & 1deg (4 corner gliders)
                  - symmetric 3-cell array centred at -1/0/+1
  C  density    fixed centre (0.5N), gliders-per-cell swept 2/4/6 at each width
  D  symhex/    symmetric REGULAR shapes (hexagon/diamond/square) with vertices on a
     symdia/    common truth circle of radius R = diameter/2: diameters 0.3/0.5/0.75/
     symsq      1.0deg centred at 0.0/0.5/-0.5N, no moorings. Truth = the disk of the
                circle (same for all three shapes at a given diameter & centre).

Note: `_equator_hex_cell` is regular only when off = sqrt(3)/2 * half_height, which no
swept width hits — so every equator_hex* config is a stretched (2-fold) hexagon. A
one-off test of the two regular hexagons found no material W-skill difference once
footprint size is matched, so they are deliberately NOT part of this config grid.

Run:  python generate_configs.py   ->   experiment_1/configs/**/*.json
"""
import json
import os

LON = 220.0                                    # 140W mooring line
MOORING_LATS = (-2.0, -1.0, 0.0, 1.0, 2.0)     # real TAO moorings on the LON line
WIDTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]      # glider longitude offset (deg)

# Symmetric REGULAR-shape families (hexagon / diamond / square). All three place
# their vertices ON a common "truth circle" of radius R = diameter/2 about the cell
# centre — the stationary analogue of the circling-glider arrays (experiment_3),
# whose gliders orbit ON the circle. Because the vertices all lie on the same circle,
# the model-truth footprint is the DISK of that circle: identical for the three
# shapes at a given diameter & centre, varying only with centre and radius (NOT with
# shape). 'diameter' d = circle diameter = 2R. No moorings (evaluated on the shape
# gliders alone). This replaces the earlier convex-hull-of-the-polygon truth.
SYMHEX_DIAMS   = [0.3, 0.5, 0.75, 1.0]         # circle diameter (deg) == 2 * radius
SYMHEX_CENTERS = [0.0, 0.5, -0.5]              # cell centre latitudes (deg N)

SYMSHAPE_DIAMS   = [0.3, 0.5, 0.75, 1.0]
SYMSHAPE_CENTERS = [0.0, 0.5, -0.5]            # shape comparison at the symhex centres
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


def _interior_moorings(positions):
    """Real moorings (integer lat on the LON line) strictly inside a cell's
    meridional span. These are interior to the convex hull, so they leave the
    hull-mean truth unchanged but add a constraint to the plane-fit estimate."""
    on_line = [p[0] for p in positions if abs(p[1] - LON) < 1e-9]
    if len(on_line) < 2:
        return []
    lo, hi = min(on_line), max(on_line)
    have = {(round(p[0], 6), round(p[1], 6)) for p in positions}
    return [[ml, LON] for ml in MOORING_LATS
            if lo + 1e-9 < ml < hi - 1e-9
            and (round(ml, 6), round(LON, 6)) not in have]


def _with_interior_moorings(cells):
    """Append any interior mooring points to every cell's position list."""
    for c in cells:
        c['positions'] = c['positions'] + _interior_moorings(c['positions'])
    return cells


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


def _equator_cell(off, half_height):
    """Single equator-centred diamond; N/S at +/-half_height, E/W gliders at +/-off.
    The centre mooring (0, LON) is supplied by _interior_moorings."""
    return [{'center_lat': 0.0,
             'positions': [[half_height, LON], [-half_height, LON],
                           [0.0, LON - off], [0.0, LON + off]]}]


def _equator_hex_cell(off, half_height):
    """Single equator-centred hexagon; N/S vertices on the LON line at +/-half,
    plus 4 side gliders at +/-off longitude and +/-half/2 latitude. The centre
    mooring (0, LON) is supplied by _interior_moorings."""
    mid = half_height / 2.0
    return [{'center_lat': 0.0,
             'positions': [[half_height, LON], [-half_height, LON],
                           [mid, LON - off], [mid, LON + off],
                           [-mid, LON - off], [-mid, LON + off]]}]


def _shift_hex_cells(centers, off):
    """The shift array as hexagons instead of diamonds: 1deg-tall hexagons centred
    at `centers`. N/S vertices sit on the LON line at center +/-0.5 (real moorings
    for the integer-straddling shift centers, so free), plus 4 side gliders at
    +/-off longitude and +/-0.25 latitude. Same 4-cell framework as _shift_cells,
    swapping the E/W diamond gliders for the hexagon's 4 side gliders (more du/dx
    info at the cost of far more gliders than the 6-glider field budget)."""
    mid = 0.25                                     # (center +/- 0.5) / 2
    return [{'center_lat': c,
             'positions': [[c + 0.5, LON], [c - 0.5, LON],
                           [c + mid, LON - off], [c + mid, LON + off],
                           [c - mid, LON - off], [c - mid, LON + off]]}
            for c in centers]


def _equator_square_cell(off, half_height):
    """Single equator-centred axis-aligned box (the diamond rotated to a square):
    4 corner gliders at +/-half_height lat and +/-off lon. Unlike the diamond,
    no corner sits on the LON line, so we add every real mooring on the line
    within the box's latitude span as a free sample point (the centre for 1deg;
    centre plus the +/-1 N/S moorings for 2deg) - the same moorings the matching
    diamond uses. Sampling U at both +/-half latitudes gives extra du/dx info."""
    corners = [[half_height, LON - off], [half_height, LON + off],
               [-half_height, LON - off], [-half_height, LON + off]]
    moor = [[ml, LON] for ml in MOORING_LATS
            if -half_height - 1e-9 <= ml <= half_height + 1e-9]
    return [{'center_lat': 0.0, 'positions': corners + moor}]


def _equator_3cell(off):
    """Symmetric no-shift array: 2deg-tall diamonds centred at -1/0/+1. Each cell's
    centre mooring is supplied by _interior_moorings."""
    return [{'center_lat': c,
             'positions': [[c - 1.0, LON], [c + 1.0, LON],
                           [c, LON - off], [c, LON + off]]}
            for c in (-1.0, 0.0, 1.0)]


def _sym_hex_cell(center, diameter):
    """Single REGULAR pointy-top hexagon whose 6 vertices lie on the truth circle of
    radius R = diameter/2 centred at `center` lat on the LON line. N/S vertices at
    +/-R (on the LON line), the 4 side gliders at +/-R/2 lat and +/-R*sqrt(3)/2 lon,
    so every vertex is exactly R from the centre. No moorings are added."""
    r = diameter / 2.0
    mid = r / 2.0                               # lat offset of the 4 side vertices
    side = r * 3.0 ** 0.5 / 2.0                 # lon offset of the 4 side vertices
    return [{'center_lat': center,
             'positions': [[center + r, LON], [center - r, LON],
                           [center + mid, LON - side], [center + mid, LON + side],
                           [center - mid, LON - side], [center - mid, LON + side]]}]


def _sym_diamond_cell(center, diameter):
    """Single REGULAR diamond (a square rotated 45deg) whose 4 vertices lie on the
    truth circle of radius R = diameter/2 centred at `center` lat on the LON line.
    N/S vertices at +/-R on the LON line, E/W vertices at +/-R longitude on the
    centre latitude, so every vertex is exactly R from the centre. No moorings."""
    r = diameter / 2.0
    return [{'center_lat': center,
             'positions': [[center + r, LON], [center - r, LON],
                           [center, LON - r], [center, LON + r]]}]


def _sym_square_cell(center, diameter):
    """Single REGULAR axis-aligned square whose 4 corners lie on the truth circle of
    radius R = diameter/2 centred at `center` lat on the LON line. Corners sit at
    +/-R/sqrt(2) latitude and +/-R/sqrt(2) longitude, so each corner is exactly R
    from the centre (the circle is circumscribed). No moorings are added."""
    r = diameter / 2.0
    s = r / 2.0 ** 0.5                          # lat/lon offset of the 4 corners
    return [{'center_lat': center,
             'positions': [[center + s, LON - s], [center + s, LON + s],
                           [center - s, LON - s], [center - s, LON + s]]}]


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

        # A. one 4-cell shift array --------------------------------------
        shift_centers = [-1.5, -0.5, 0.5, 1.5]
        cells = _with_interior_moorings(_shift_cells(shift_centers, off))
        written.append(_write('shift', dict(
            name=f'shift_{wk}', family='shift', pattern='shift',
            width=off, n_gliders_per_cell=2, n_gliders_total=_n_gliders_total(cells),
            cell_height_deg=1.0,
            description=(f'4-cell shift array, 1deg diamonds centred at {shift_centers}, '
                         f'glider lon offset {off} deg.'),
            positions=_union_positions(cells), cells=cells)))

        # A'. the same 4-cell shift framework, but 1deg hexagons ------------
        cells = _with_interior_moorings(_shift_hex_cells(shift_centers, off))
        written.append(_write('shift', dict(
            name=f'shift_hex_{wk}', family='shift', pattern='shift_hex',
            width=off, n_gliders_per_cell=4, n_gliders_total=_n_gliders_total(cells),
            cell_height_deg=1.0,
            description=(f'4-cell shift array, 1deg hexagons centred at {shift_centers}, '
                         f'glider lon offset {off} deg (4 side gliders per cell).'),
            positions=_union_positions(cells), cells=cells)))

        # B. equator-centred ---------------------------------------------
        # single diamonds (2deg mooring-spanned, 1deg glider-spanned)
        for tag, half in (('2deg', 1.0), ('1deg', 0.5)):
            cells = _with_interior_moorings(_equator_cell(off, half))
            written.append(_write('equator', dict(
                name=f'equator_{tag}_{wk}', family='equator', pattern=f'equator_{tag}',
                width=off, n_gliders_per_cell=_n_gliders_total(cells),
                n_gliders_total=_n_gliders_total(cells), cell_height_deg=2 * half,
                description=(f'Equator-centred diamond, {2 * half:g}deg tall, glider lon '
                             f'offset {off} deg, with centre mooring.'),
                positions=_union_positions(cells), cells=cells)))

        # single hexagons (2deg -> 4 gliders, 1deg -> 6 gliders)
        for tag, half, ng in (('hex2deg', 1.0, 4), ('hex1deg', 0.5, 6)):
            cells = _with_interior_moorings(_equator_hex_cell(off, half))
            written.append(_write('equator', dict(
                name=f'equator_{tag}_{wk}', family='equator', pattern=f'equator_{tag}',
                width=off, n_gliders_per_cell=ng, n_gliders_total=ng,
                cell_height_deg=2 * half,
                description=(f'Equator-centred hexagon, {2 * half:g}deg tall, {ng} gliders, '
                             f'glider lon offset {off} deg, with centre mooring.'),
                positions=_union_positions(cells), cells=cells)))

        # single squares (diamond rotated to a box; 4 corner gliders)
        for tag, half in (('sq2deg', 1.0), ('sq1deg', 0.5)):
            cells = _with_interior_moorings(_equator_square_cell(off, half))
            written.append(_write('equator', dict(
                name=f'equator_{tag}_{wk}', family='equator', pattern=f'equator_{tag}',
                width=off, n_gliders_per_cell=_n_gliders_total(cells),
                n_gliders_total=_n_gliders_total(cells), cell_height_deg=2 * half,
                description=(f'Equator-centred square (box), {2 * half:g}deg tall, 4 corner '
                             f'gliders at glider lon offset {off} deg, with span moorings.'),
                positions=_union_positions(cells), cells=cells)))

        # symmetric 3-cell array
        cells = _with_interior_moorings(_equator_3cell(off))
        written.append(_write('equator', dict(
            name=f'equator_3cell_{wk}', family='equator', pattern='equator_3cell',
            width=off, n_gliders_per_cell=2, n_gliders_total=_n_gliders_total(cells),
            cell_height_deg=2.0,
            description=(f'Symmetric no-shift 3-cell array, 2deg-tall diamonds centred at '
                         f'-1/0/+1, glider lon offset {off} deg, with centre moorings.'),
            positions=_union_positions(cells), cells=cells)))

        # C. density sweep at fixed centre (0.5N) ------------------------
        for ng, rows in ((2, [0.5]), (4, [0.25, 0.75]), (6, [0.25, 0.5, 0.75])):
            cells = _with_interior_moorings(_density_cell(off, rows))
            written.append(_write('density', dict(
                name=f'density_{ng}g_{wk}', family='density', pattern=f'density_{ng}g',
                width=off, n_gliders_per_cell=ng, n_gliders_total=ng,
                cell_height_deg=1.0,
                description=(f'Fixed centre 0.5N, {ng} gliders/cell (rows {rows}), '
                             f'glider lon offset {off} deg.'),
                positions=_union_positions(cells), cells=cells)))

    # D. symmetric REGULAR-hexagon sweep (diameter x centre, no moorings) --
    #    Vertices lie on the truth circle of radius R = diameter/2; the model-truth
    #    footprint is that disk (identical across shapes at a given diameter/centre).
    for d in SYMHEX_DIAMS:
        r = d / 2.0
        for c in SYMHEX_CENTERS:
            cells = _sym_hex_cell(c, d)            # no _with_interior_moorings: no moorings
            written.append(_write('symhex', dict(
                name=f'symhex_d{d}_c{c:+.1f}', family='symhex', pattern=f'symhex_d{d}',
                width=r, diameter=d, radius=r, center_lat=c, center_lon=LON,
                n_gliders_per_cell=6, n_gliders_total=6,
                cell_height_deg=2 * r,             # N-S vertex-to-vertex extent = 2R = d
                description=(f'Symmetric REGULAR hexagon, 6 vertices on the truth circle '
                             f'of radius {r:g}deg (diameter {d:g}deg), centred at '
                             f'{c:+g}N/{LON:g}E, no moorings.'),
                positions=_union_positions(cells), cells=cells)))

    # D2. symmetric REGULAR square / diamond sweeps (the shape analogues of symhex,
    #     same circle diameters, 140W, no moorings) --------------------------
    for d in SYMSHAPE_DIAMS:
        r = d / 2.0
        for c in SYMSHAPE_CENTERS:
            for subdir, fam, builder, shp, height in (
                    ('symdia', 'symdia', _sym_diamond_cell, 'diamond', 2 * r),
                    ('symsq',  'symsq',  _sym_square_cell,  'square', 2 * r / 2.0 ** 0.5)):
                cells = builder(c, d)              # no moorings, like symhex
                written.append(_write(subdir, dict(
                    name=f'{fam}_d{d}_c{c:+.1f}', family=fam, pattern=f'{fam}_d{d}',
                    width=r, diameter=d, radius=r, center_lat=c, center_lon=LON,
                    n_gliders_per_cell=4, n_gliders_total=4,
                    cell_height_deg=height,
                    description=(f'Symmetric REGULAR {shp}, 4 vertices on the truth circle '
                                 f'of radius {r:g}deg (diameter {d:g}deg), centred at '
                                 f'{c:+g}N/{LON:g}E, no moorings.'),
                    positions=_union_positions(cells), cells=cells)))
    return written


if __name__ == '__main__':
    files = build()
    print(f'Wrote {len(files)} configs under {CFG_ROOT}')
    for f in files:
        print('  ', f)
