"""Anisotropic potential field and voxel visibility (paper section 3.2)."""
import numpy as np
from scipy.spatial import cKDTree


def potential_field(grid, interior):
    """L2 distance from each interior voxel to the nearest busy voxel in the
    upper half-space (z' >= z), in meters."""
    busy_idx = np.argwhere(grid.busy)
    order = np.argsort(busy_idx[:, 2])[::-1]  # descending z
    busy_idx = busy_idx[order]
    pf = np.zeros(grid.shape)
    for z in range(grid.shape[2]):
        cells = np.argwhere(interior[:, :, z])
        upper = busy_idx[busy_idx[:, 2] >= z]
        if cells.size and upper.size:
            q = np.column_stack([cells, np.full(len(cells), z)])
            d, _ = cKDTree(upper).query(q)
            pf[cells[:, 0], cells[:, 1], z] = d * grid.voxel
    return pf


def pf_map(pf, interior):
    """Per-stack max PF (2D) and index of the highest interior voxel (-1 = none)."""
    nz = interior.shape[2]
    top_z = np.where(interior.any(2), nz - 1 - np.argmax(interior[:, :, ::-1], 2), -1)
    return pf.max(axis=2), top_z


def visibility(grid, top_z, cells, targets=None, step=0.5, chunk=8):
    """Mutual visibility between the highest interior voxels of the given stacks.

    Returns bool (S, T): True where the segment between voxel centers crosses
    no busy voxel, sampled every `step` voxels. targets selects a column
    subset of cells (default: all)."""
    p = (np.column_stack([cells, top_z[tuple(cells.T)]]) + 0.5).astype(np.float32)
    q = p if targets is None else p[targets]
    k = int(np.ceil(np.linalg.norm(grid.shape) / step))
    t = ((np.arange(k, dtype=np.float32) + 0.5) / k)[None, None, :, None]
    vis = np.empty((len(p), len(q)), bool)
    for s in range(0, len(p), chunk):
        a = p[s:s + chunk, None, None, :]
        smp = (a + t * (q[None, :, None, :] - a)).astype(np.int64)  # (m, T, k, 3)
        vis[s:s + chunk] = ~grid.busy[smp[..., 0], smp[..., 1], smp[..., 2]].any(2)
    return vis
