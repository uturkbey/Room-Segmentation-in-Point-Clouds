"""Room clustering on the 2D PF map (paper section 3.3)."""
import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN


def histogram_thresholds(pf_vals, bins=32):
    """(thr_low, thr_high): PF values of the lowest and highest histogram density peaks."""
    counts, edges = np.histogram(pf_vals, bins)
    mids = (edges[:-1] + edges[1:]) / 2
    peaks, _ = find_peaks(counts)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(counts))])
    return mids[peaks[0]], mids[peaks[-1]]


def distance_matrix(cells, pf_vals, vis, w=(0.3, 0.6, 0.1)):
    """Pairwise distance D per Eq. (2): weighted visibility + euclidean + PF terms."""
    v = vis.astype(np.float32)
    hamming = v @ (1 - v).T + (1 - v) @ v.T
    d_vis = hamming / np.maximum(v.sum(1)[:, None] + v.sum(1)[None, :], 1)  # Eq. (3)
    d_eucl = squareform(pdist(cells.astype(float)))
    d_pf = np.abs(pf_vals[:, None] - pf_vals[None, :])
    return (w[0] * d_vis
            + w[1] * d_eucl / max(d_eucl.max(), 1e-9)
            + w[2] * d_pf / max(d_pf.max(), 1e-9))


def cluster_rooms(pf_vals, D, thr_low, thr_high, min_cluster_size=8, new_cluster_dist=0.3):
    """HDBSCAN on high-PF seed cells, then nearest-cluster assignment of the rest.

    Returns per-cell labels; -1 = below thr_low (unassigned)."""
    labels = np.full(len(pf_vals), -1)
    seeds = pf_vals >= thr_high
    if seeds.sum() >= min_cluster_size:
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
        with warnings.catch_warnings():  # sklearn's `copy` deprecation; D subset is a copy already
            warnings.simplefilter("ignore", FutureWarning)
            labels[seeds] = model.fit_predict(D[np.ix_(seeds, seeds)].astype(np.float64))

    rest = np.where((pf_vals >= thr_low) & (labels < 0))[0]
    nxt = labels.max() + 1
    for i in rest[np.argsort(-pf_vals[rest])]:  # strongest cells first
        assigned = labels >= 0
        d = D[i, assigned]
        if d.size and d.min() <= new_cluster_dist:
            labels[i] = labels[assigned][d.argmin()]
        else:
            labels[i] = nxt
            nxt += 1
    return labels
