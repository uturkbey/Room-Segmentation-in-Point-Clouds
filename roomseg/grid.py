"""Occupancy voxel grid, Z vertical."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    origin: np.ndarray  # (3,) world position of voxel (0,0,0) corner
    voxel: float
    busy: np.ndarray  # bool (nx, ny, nz)

    @property
    def shape(self):
        return self.busy.shape

    def index_of(self, xyz):
        """World points -> integer voxel indices, clipped to the grid."""
        i = np.floor((xyz - self.origin) / self.voxel).astype(np.int64)
        return np.clip(i, 0, np.array(self.shape) - 1)

    def centers(self, idx):
        """Voxel indices -> world coordinates of voxel centers."""
        return self.origin + (np.asarray(idx) + 0.5) * self.voxel


def voxelize(xyz, voxel=0.18):
    """Bounding-box grid; a voxel is busy iff it contains at least one point."""
    origin = xyz.min(axis=0)
    shape = np.floor((xyz.max(axis=0) - origin) / voxel).astype(np.int64) + 1
    grid = Grid(origin, voxel, np.zeros(shape, bool))
    grid.busy[tuple(grid.index_of(xyz).T)] = True
    return grid
