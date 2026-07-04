"""Interior free-space classification (paper section 3.1)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


def remove_clutter(busy):
    """Free busy voxels matching the vertical busy-free-BUSY-free-busy pattern."""
    below = np.maximum.accumulate(busy, axis=2)  # busy anywhere at z' <= z
    above = np.maximum.accumulate(busy[:, :, ::-1], axis=2)[:, :, ::-1]
    freed = np.zeros_like(busy)
    freed[:, :, 2:-2] = (busy[:, :, 2:-2] & ~busy[:, :, 1:-3] & ~busy[:, :, 3:-1]
                         & below[:, :, :-4] & above[:, :, 4:])
    return busy & ~freed


def dominant_directions(normals):
    """Two dominant horizontal axes from wall normals (axial mean folded mod 90deg)."""
    n = normals[np.abs(normals[:, 2]) < 0.3]
    theta = np.arctan2(n[:, 1], n[:, 0])
    a = np.angle(np.exp(4j * theta).mean()) / 4
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def _blocked(busy, idx, d, chunk=4096):
    """True per voxel in idx if a busy voxel lies along horizontal direction d."""
    nx, ny, _ = busy.shape
    steps = 0.5 * np.arange(1, 2 * max(nx, ny) + 1)
    out = np.zeros(len(idx), bool)
    for s in range(0, len(idx), chunk):
        p = idx[s:s + chunk, None, :2] + 0.5 + steps[None, :, None] * d  # (m, K, 2)
        q = np.floor(p).astype(np.int64)
        inb = (q >= 0).all(-1) & (q[..., 0] < nx) & (q[..., 1] < ny)
        q[~inb] = 0
        out[s:s + chunk] = (busy[q[..., 0], q[..., 1], idx[s:s + chunk, None, 2]] & inb).any(1)
    return out


def evidence(busy, doms):
    """Interior evidence E(v) per Eq. (1); zero outside free space."""
    cum = np.maximum.accumulate(busy, axis=2)
    rcm = np.maximum.accumulate(busy[:, :, ::-1], axis=2)[:, :, ::-1]
    below = np.zeros_like(busy)
    above = np.zeros_like(busy)
    below[:, :, 1:] = cum[:, :, :-1]  # busy strictly below
    above[:, :, :-1] = rcm[:, :, 1:]  # busy strictly above
    E = 0.43 * below + 0.1425 * (above + (above & below))
    idx = np.argwhere(~busy)
    for d in doms:
        E[tuple(idx.T)] += 0.1425 * (_blocked(busy, idx, d) & _blocked(busy, idx, -d))
    E[busy] = 0.0
    return E


def segment_interior(E, busy, smoothness=0.6, scale=1000):
    """MRF over free voxels solved by min-cut; returns bool interior mask."""
    free = ~busy
    n = int(free.sum())
    node = np.full(busy.shape, -1, np.int64)
    node[free] = np.arange(2, n + 2)  # 0 = source (interior), 1 = sink (exterior)
    e = np.rint(E[free] * scale).astype(np.int32)

    R = [np.zeros(n, np.int64), node[free]]
    C = [node[free], np.ones(n, np.int64)]
    V = [e, scale - e]
    w = np.int32(round(smoothness * scale))
    for ax in range(3):
        lo = tuple(slice(None, -1) if i == ax else slice(None) for i in range(3))
        hi = tuple(slice(1, None) if i == ax else slice(None) for i in range(3))
        a, b = node[lo], node[hi]
        m = (a >= 2) & (b >= 2)
        a, b = a[m], b[m]
        R += [a, b]
        C += [b, a]
        V += [np.full(a.size, w), np.full(b.size, w)]

    g = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                   shape=(n + 2, n + 2), dtype=np.int32).tocsr()
    residual = g - maximum_flow(g, 0, 1).flow
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, 0, return_predecessors=False)

    on_source_side = np.zeros(n + 2, bool)
    on_source_side[reachable] = True
    interior = np.zeros_like(busy)
    interior[free] = on_source_side[node[free]]
    return interior
