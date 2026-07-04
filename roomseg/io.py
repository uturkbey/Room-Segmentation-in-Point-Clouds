"""PLY I/O. Internal convention is Z-up; Y-up data is swapped on load/save."""
import numpy as np
from plyfile import PlyData, PlyElement

_ORDER = {"y": [0, 2, 1], "z": [0, 1, 2]}  # self-inverse permutations


def load_ply(path, up="y"):
    """Read a point cloud; return (xyz, normals) as float64 (N, 3), Z-up."""
    v = PlyData.read(str(path))["vertex"].data
    o = _ORDER[up]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    nrm = np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float64)
    return xyz[:, o], nrm[:, o]


def save_labeled_ply(path, xyz, normals, labels, up="y"):
    """Write x, y, z, nx, ny, nz + int room label (-1 = unlabeled), original up-axis."""
    o = _ORDER[up]
    cols = np.hstack([xyz[:, o], normals[:, o]]).astype(np.float32)
    names = ("x", "y", "z", "nx", "ny", "nz")
    arr = np.empty(len(xyz), dtype=[(n, "f4") for n in names] + [("label", "i4")])
    for i, n in enumerate(names):
        arr[n] = cols[:, i]
    arr["label"] = labels
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
