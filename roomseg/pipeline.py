"""End-to-end room segmentation (paper Fig. 2) with per-stage benchmarks."""
import resource
import sys
import time
from contextlib import contextmanager

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode

from .cluster import cluster_rooms, distance_matrix, histogram_thresholds
from .field import pf_map, potential_field, visibility
from .grid import voxelize
from .interior import dominant_directions, evidence, remove_clutter, segment_interior

_RSS_DIV = 2**20 if sys.platform == "darwin" else 2**10  # ru_maxrss unit -> MB


@contextmanager
def _timed(name, bench):
    t0 = time.perf_counter()
    yield
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIV
    bench.append((name, time.perf_counter() - t0, rss))


def segment(xyz, normals, voxel=0.18, smoothness=0.6, vis_targets=None,
            min_cluster_size=8, new_cluster_dist=0.3, knn=10, rng_seed=0):
    """Segment a Z-up point cloud into rooms.

    Returns (labels, art): per-point room labels (-1 = unassigned) and a dict of
    intermediate artifacts including per-stage (name, seconds, peak-RSS-MB)."""
    bench = []
    with _timed("voxelize", bench):
        grid = voxelize(xyz, voxel)
        grid.busy = remove_clutter(grid.busy)
    with _timed("interior MRF", bench):
        E = evidence(grid.busy, dominant_directions(normals))
        interior = segment_interior(E, grid.busy, smoothness)
    with _timed("potential field", bench):
        pf = potential_field(grid, interior)
        pf2d, top_z = pf_map(pf, interior)
    with _timed("visibility", bench):
        cells = np.argwhere(top_z >= 0)
        targets = None
        if vis_targets and vis_targets < len(cells):
            targets = np.random.default_rng(rng_seed).choice(len(cells), vis_targets, False)
        vis = visibility(grid, top_z, cells, targets)
    with _timed("clustering", bench):
        pf_vals = pf2d[tuple(cells.T)]
        thr_low, thr_high = histogram_thresholds(pf_vals)
        D = distance_matrix(cells, pf_vals, vis)
        cell_labels = cluster_rooms(pf_vals, D, thr_low, thr_high,
                                    min_cluster_size, new_cluster_dist)
    with _timed("back-projection", bench):
        labels = _backproject(grid, interior, cells, cell_labels, xyz, knn)

    art = dict(grid=grid, interior=interior, pf=pf, pf2d=pf2d, cells=cells,
               cell_labels=cell_labels, thresholds=(thr_low, thr_high),
               n_rooms=int(cell_labels.max()) + 1, bench=bench)
    return labels, art


def _backproject(grid, interior, cells, cell_labels, xyz, knn):
    """Paper section 3.4: stack propagation, then kNN majority vote for the rest."""
    lab3 = np.full(grid.shape, -1)
    for (x, y), lab in zip(cells, cell_labels):
        if lab >= 0:
            lab3[x, y, interior[x, y]] = lab
    labeled = np.argwhere(lab3 >= 0)

    pv = grid.index_of(xyz)
    uniq, inv = np.unique(pv, axis=0, return_inverse=True)
    need = lab3[tuple(uniq.T)] < 0
    if need.any() and len(labeled):
        k = min(knn, len(labeled))
        nn = cKDTree(labeled).query(uniq[need], k=k)[1].reshape(-1, k)
        votes = lab3[tuple(labeled[nn].transpose(2, 0, 1))]
        lab3[tuple(uniq[need].T)] = mode(votes, axis=1).mode
    return lab3[tuple(uniq.T)][inv]
