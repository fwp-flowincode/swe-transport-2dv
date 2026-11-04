"""Digitize the red measurement circles of the splitter-plate profile plots (Delft3D-FLOW Validation Document 2008).

Usage: python experiments/splitter_plate/digitize_measurements.py [review folder]
Reads the three screenshots of the profile figures from measurements/ (kept local, not in the repository) and writes
data/velocity.csv, data/relative_density.csv and data/tke.csv; with a review folder also enlarged crops with a cross at
every centre, for checking the result by eye. Per image four panels (x = 2, 5, 10, 40 m). Calibration: the frame of each
panel spans the axis range; the inward ticks give the scale. Circle centres: correlation of the red pixels with a hollow
ring of radius RING_R (+ on the ring, - in the centre), local maxima above SCORE_MIN; then the manual corrections below
(points to drop and to add, in data coordinates; an added point is moved to the best ring fit within SNAP_PX).
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "measurements")
OUT = os.path.join(HERE, "data")
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else None
STATIONS = (2, 5, 10, 40)
RING_R = 6.0          # px, radius of the circles (measured on an isolated circle: mean distance 6.1 px)
SCORE_MIN = 0.25      # -, ring score threshold of the candidates (1: complete red ring with an empty centre)
SECTORS = 24          # -, angular sectors of the ring completeness test
RED_COVER_MIN = 0.6   # -, share of the sectors with red on the ring
ANY_COVER_MIN = 0.8   # -, share with red or blue (the model line is drawn over the rings and hides parts of them)
RING_BAND = 1.5       # px, half width of the ring band
MIN_DIST = 4          # px, smallest distance of two detected centres
SNAP_PX = 2           # px, search radius of the ring fit for added points (clustered circles lie 5 px apart)
IMAGES = {
    # lefts/rights: frame columns of the panels; x_zero: offset of the data zero from the left frame (px); per_px: data
    # per pixel in x; y_zero: row of z = 0; px_per_m: rows per metre (ticks of the first panel; the panels share them)
    "velocity": dict(file="velocity_profiles.png", lefts=(42, 248.5, 455.5, 662), rights=(192, 399, 606, 811),
                     top=44, bottom=480, x_zero=-1.0, per_px=0.2 / 50.5, y_zero=478.5, px_per_m=725.0,
                     quantity="u", unit="m/s"),
    "relative_density": dict(file="relative_density_profiles.png", lefts=(44, 250.5, 457.5, 664),
                             rights=(200.5, 407, 614, 821), top=39, bottom=474, x_zero=-1.5, per_px=0.5 / 65.5,
                             y_zero=473.0, px_per_m=725.0, quantity="relative_density", unit="-"),
    "tke": dict(file="tke_profiles.png", lefts=(45, 251.5, 458.5, 665), rights=(184, 391, 598, 803), top=44,
                bottom=471, x_zero=0.0, per_px=0.001 / 68.5, y_zero=469.4, px_per_m=708.6, quantity="k",
                unit="m^2/s^2"),
}
# manual corrections after review of the overlays: station (m) -> list of (value, z) to drop / to add
CORRECTIONS = {name: {"drop": {}, "add": {}} for name in IMAGES}
# relative density at 40 m: circles on the left frame (cut by it) and the overlapping cluster at z = 0.29 m
CORRECTIONS["relative_density"]["add"][40] = [(0.0, 0.49), (0.0, 0.39), (0.04, 0.35), (0.38, 0.29), (0.43, 0.29),
                                              (0.47, 0.29), (0.52, 0.29), (0.57, 0.285)]
# velocity at 40 m: a circle whose ring the model line crosses twice
CORRECTIONS["velocity"]["add"][40] = [(0.33, 0.124)]


def ring_score(red):
    R = int(np.ceil(RING_R + 2))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(yy, xx)
    ring = (np.abs(d - RING_R) <= 1.2).astype(float)
    hole = (d <= RING_R - 2.5).astype(float)
    kern = ring / ring.sum() - hole / hole.sum()
    return fftconvolve(red.astype(float), kern[::-1, ::-1], mode="same")


def coverage(mask, x, y):
    H, W = mask.shape
    hit = 0
    for t in np.linspace(0, 2 * np.pi, SECTORS, endpoint=False):
        ok = False
        for rr in np.arange(RING_R - RING_BAND, RING_R + RING_BAND + 0.01, 0.5):
            for dt in (-0.06, 0.0, 0.06):
                yy, xx = int(round(y + rr * np.sin(t + dt))), int(round(x + rr * np.cos(t + dt)))
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx]:
                    ok = True
        hit += ok
    return hit / SECTORS


def to_data(c, p, x, y):
    return (x - (c["lefts"][p] + c["x_zero"])) * c["per_px"], (c["y_zero"] - y) / c["px_per_m"]


def to_pixel(c, p, v, z):
    return v / c["per_px"] + c["lefts"][p] + c["x_zero"], c["y_zero"] - z * c["px_per_m"]


for name, c in IMAGES.items():
    im = Image.open(os.path.join(SOURCE, c["file"])).convert("RGB")
    a = np.asarray(im).astype(int)
    red = (a[..., 0] - (a[..., 1] + a[..., 2]) / 2) > 50
    blue = (a[..., 2] - (a[..., 0] + a[..., 1]) / 2) > 50
    score = ring_score(red)
    peak = (score == ndimage.maximum_filter(score, size=2 * MIN_DIST + 1)) & (score > SCORE_MIN)
    found = {p: [] for p in range(4)}
    for y, x in zip(*np.nonzero(peak)):
        p = next((p for p in range(4) if c["lefts"][p] - 3 <= x <= c["rights"][p] + 3), None)
        if p is not None and c["top"] - 3 <= y <= c["bottom"] + 3:
            if coverage(red, x, y) >= RED_COVER_MIN and coverage(red | blue, x, y) >= ANY_COVER_MIN:
                found[p].append((float(x), float(y)))
    corr = CORRECTIONS[name]
    for p, s in enumerate(STATIONS):
        for v, z in corr["drop"].get(s, []):
            px, py = to_pixel(c, p, v, z)
            j = int(np.argmin([np.hypot(x - px, y - py) for x, y in found[p]]))
            found[p].pop(j)
        for v, z in corr["add"].get(s, []):
            px, py = to_pixel(c, p, v, z)
            x0, y0 = int(round(px)), int(round(py))
            win = score[y0 - SNAP_PX:y0 + SNAP_PX + 1, x0 - SNAP_PX:x0 + SNAP_PX + 1]
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            found[p].append((float(x0 - SNAP_PX + dx), float(y0 - SNAP_PX + dy)))
    rows = sorted((s, *reversed(to_data(c, p, x, y))) for p, s in enumerate(STATIONS) for x, y in found[p])
    # review crops: green cross at every centre, enlarged, with a data grid every 0.05 m in z
    scale = 3
    if SCRATCH is not None:
        big = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        d = ImageDraw.Draw(big)
        for p in range(4):
            for zg in np.arange(0, 0.601, 0.05):
                _, yg = to_pixel(c, p, 0, zg)
                d.line((c["lefts"][p] * scale, yg * scale, c["rights"][p] * scale, yg * scale), fill=(200, 200, 200), width=1)
            for x, y in found[p]:
                X, Y = x * scale + 1, y * scale + 1
                d.line((X - 5, Y, X + 5, Y), fill=(0, 170, 0), width=2)
                d.line((X, Y - 5, X, Y + 5), fill=(0, 170, 0), width=2)
        for p in range(4):
            box = (int((c["lefts"][p] - 40) * scale), int((c["top"] - 5) * scale), int((c["rights"][p] + 8) * scale),
                   int((c["bottom"] + 25) * scale))
            big.crop(box).save(f"{SCRATCH}/review_{name}_{STATIONS[p]}m.png")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="\r\n") as fh:
        fh.write(f"# Splitter plate: measured {c['quantity']} ({c['unit']}) against the height above the bed z (m), at x (m)\n")
        fh.write("# behind the end of the plate. Extracted visually (digitized from the plotted circles) from the profile\n")
        fh.write("# figures of the Delft3D-FLOW Validation Document (2008), splitter plate test; not the original data.\n")
        fh.write(f"# Accuracy of the extraction about one pixel: {c['per_px']:.2g} {c['unit']} and {1 / c['px_per_m']:.1g} m.\n")
        fh.write("# Overlapping circles may be merged into one point. Digitizer: experiments/splitter_plate/digitize_measurements.py\n")
        fh.write(f"x_m,z_m,{c['quantity']}\n")
        for s, z, v in rows:
            fh.write(f"{s},{z:.4f},{v:.5g}\n")
    print(name, "points per station", [len(found[p]) for p in range(4)])
