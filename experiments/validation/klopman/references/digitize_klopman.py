# Digitise the marker plots of Klopman (1994) by template matching.
# usage: digitize.py page.png x0 dx z0 dz [logz] [tx ty] [thr]
#  grid columns inside the box get x values x0, x0+dx, ...; grid rows from the bottom get z0, z0+dz, ... (log: decades)
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
path = sys.argv[1]; x0, dx, z0, dz = map(float, sys.argv[2:6])
logz = len(sys.argv) > 6 and sys.argv[6] == 'log'
img = 1.0 - np.array(Image.open(path).convert('L')) / 255.0
a = img > 0.5
H, W = a.shape
def lines(axis, frac):
    b = (img > 0.37)[300:3000, 300:2300]
    s = b.sum(axis=axis)
    idx = np.where(s > 0.25 * s.max())[0]
    groups = np.split(idx, np.where(np.diff(idx) > 8)[0] + 1)
    return [(g[0] + g[-1]) / 2 + 300 for g in groups if len(g)]
rows = lines(1, 0)[:5 if logz else 6]; cols = lines(0, 0)[:7]
print('cols', [round(c) for c in cols], 'rows', [round(r) for r in rows], file=sys.stderr)
# template: isolated marker given as tx ty
tx, ty = int(sys.argv[7]), int(sys.argv[8])
thr = float(sys.argv[9]) if len(sys.argv) > 9 else 0.6
r = 10
timg = 1.0 - np.array(Image.open('klopman/page68.png').convert('L')) / 255.0
t = timg[ty - r:ty + r + 1, tx - r:tx + r + 1].copy()
t -= t.mean()
# normalised cross-correlation
num = fftconvolve(img, t[::-1, ::-1], mode='same')
loc = fftconvolve(img, np.ones_like(t), mode='same'); loc2 = fftconvolve(img**2, np.ones_like(t), mode='same')
var = np.maximum(loc2 - loc**2 / t.size, 1e-9)
ncc = num / np.sqrt(var * (t**2).sum())
mx = ndi.maximum_filter(ncc, size=9)
pk = np.argwhere((ncc == mx) & (ncc > thr))
c0, c1 = cols[0], cols[-1]; r0, r1 = rows[0], rows[-1]
out = []
for (y, x) in pk:
    if not (c0 - 5 <= x <= c1 + 5 and r0 - 5 <= y <= r1 + 5): continue
    xv = x0 + dx * (x - c0) / ((c1 - c0) / (len(cols) - 1))
    zz = (r1 - y) / ((r1 - r0) / (len(rows) - 1))
    zv = z0 * 10**(zz * dz) if logz else z0 + dz * zz
    out.append((zv, xv, ncc[y, x], x, y))
out.sort()
ded=[]
for o in out:
    if all(abs(o[3]-d[3])>4 or abs(o[4]-d[4])>4 for d in ded): ded.append(o)
out=ded
for o in out: print('%.5f %.4f %.2f %d %d' % o)
