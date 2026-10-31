"""Offline-only occupied-cell rectangle partition;not wired into any runtime.

Merge identical horizontal occupied runs across adjacent rows. Their closed
union is exactly the original cell union: no road expansion or hole filling.
The existing body SAT and scalar near-contact fallback remain unchanged.
"""
import numpy as np
from navstack.road_grid_batch import ExactRoadBatch


def occupied_partition(drivable):
    mask=np.asarray(drivable)
    if mask.ndim!=2 or mask.dtype!=np.bool_:
        raise ValueError('Require a two-dimensional boolean drivable mask')
    active={};finished=[]
    for y,row in enumerate(~mask):
        edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
        runs=list(zip(edges[::2].tolist(),edges[1::2].tolist()))
        present=set(runs)
        for key in list(active):
            if key not in present:finished.append(active.pop(key))
        for left,right in runs:
            key=(left,right)
            if key in active:active[key][3]=y+1
            else:active[key]=[left,y,right,y+1]
    finished.extend(active.values())
    return np.asarray(finished,dtype=np.int64).reshape(-1,4)


class RectangleRoadBatch(ExactRoadBatch):
    """Experimental partition only;all collision equations inherit production."""
    def _refresh(self):
        self.mask=self.grid.collision_drivable.copy()
        self.cell_size=(self.grid.dx,self.grid.dy)
        indices=occupied_partition(self.mask)
        self.rectangles=indices*np.array([self.grid.dx,self.grid.dy,self.grid.dx,self.grid.dy])
        self.centers=(self.rectangles[:,:2]+self.rectangles[:,2:])/2
        self.radii=(self.rectangles[:,2:]-self.rectangles[:,:2])/2
