"""Opt-in full-body road SAT batching (batch_sat_v1); scalar remains default.

Closed occupied cells, full vehicle corners and the scalar oracle's tolerance
are retained. Numerically sensitive contacts use the unchanged scalar routine.
"""
import math
import numpy as np


def _body_corners_many(poses, vehicle, margin=0.):
    """Same scalar-libm rotations and matrix operations, batched allocations."""
    count = len(poses)
    cosine = np.fromiter(map(math.cos, poses[:, 2]), dtype=float, count=count)
    sine = np.fromiter(map(math.sin, poses[:, 2]), dtype=float, count=count)
    rotation = np.empty((count, 2, 2))
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = sine
    rotation[:, 1, 0] = -sine
    rotation[:, 1, 1] = cosine
    local = np.array([[-vehicle.rear-margin, -vehicle.width/2-margin],
                      [vehicle.front+margin, -vehicle.width/2-margin],
                      [vehicle.front+margin, vehicle.width/2+margin],
                      [-vehicle.rear-margin, vehicle.width/2+margin]])
    return local @ rotation + poses[:, None, :2]


class ExactRoadBatch:
    """Cache occupied cells, group work spatially, retain the scalar contact oracle."""
    def __init__(self, grid, chunk_size=128):
        if type(chunk_size) is not int or not 1<=chunk_size<=512:
            raise ValueError('Road SAT chunk size must be1..512')
        self.grid=grid;self.chunk_size=chunk_size
        self._refresh()

    def _refresh(self):
        self.mask=self.grid.drivable.copy()
        self.cell_size=(self.grid.dx,self.grid.dy)
        yy,xx=np.nonzero(~self.mask)
        self.rectangles=np.c_[xx*self.grid.dx,yy*self.grid.dy,
            (xx+1)*self.grid.dx,(yy+1)*self.grid.dy]
        self.centers=(self.rectangles[:,:2]+self.rectangles[:,2:])/2
        self.radii=(self.rectangles[:,2:]-self.rectangles[:,:2])/2

    def intersects_many(self, poses, vehicle, margin=0.):
        from .world import body_corners
        poses=np.asarray(poses,dtype=float)
        if poses.ndim!=2 or poses.shape[1]!=3 or not np.isfinite(poses).all():
            raise ValueError('Finite N x3 poses required')
        if not math.isfinite(margin) or margin<0:raise ValueError('Nonnegative finite margin required')
        if self.cell_size!=(self.grid.dx,self.grid.dy) or not np.array_equal(self.mask,self.grid.drivable):
            self._refresh()
        result=np.zeros(len(poses),dtype=bool);self.last_scalar_fallbacks=0
        if not len(poses) or not len(self.rectangles):return result
        # Preserve scalar libm and local-matrix operation order; share array
        # construction instead of allocating three matrices per pose.
        corners=_body_corners_many(poses,vehicle,margin)
        tolerance=1e-12;contact_band=1e-8
        groups=[np.arange(len(poses))]
        if len(poses)>self.chunk_size:
            # A scattered batch must not compare every body with the union of
            # occupied cells across the whole map. Coarse spatial bins only
            # group work; they never discard any potentially colliding cell.
            tiles=np.floor(poses[:,:2]/(8*np.asarray(self.cell_size))).astype(np.int64)
            _,labels=np.unique(tiles,axis=0,return_inverse=True)
            groups=[np.flatnonzero(labels==label) for label in range(int(labels.max())+1)]
        chunks=[group[begin:begin+self.chunk_size] for group in groups
                for begin in range(0,len(group),self.chunk_size)]
        for indices in chunks:
            body=corners[indices]
            lo,hi=body.min(axis=1),body.max(axis=1)
            union_lo,union_hi=lo.min(axis=0),hi.max(axis=0)
            rect=self.rectangles
            keep=((rect[:,0]<=union_hi[0]+contact_band)&(rect[:,2]>=union_lo[0]-contact_band)
                &(rect[:,1]<=union_hi[1]+contact_band)&(rect[:,3]>=union_lo[1]-contact_band))
            rect=rect[keep]
            if not len(rect):continue
            centers,radii=self.centers[keep],self.radii[keep]
            xleft=hi[:,0,None]-rect[None,:,0]+tolerance
            xright=rect[None,:,2]-lo[:,0,None]+tolerance
            ybottom=hi[:,1,None]-rect[None,:,1]+tolerance
            ytop=rect[None,:,3]-lo[:,1,None]+tolerance
            xhit=(xleft>=0)&(xright>=0);yhit=(ybottom>=0)&(ytop>=0)
            box=xhit&yhit;overlap=box.copy()
            sensitive=(((np.abs(xleft)<contact_band)|(np.abs(xright)<contact_band))&yhit
                |((np.abs(ybottom)<contact_band)|(np.abs(ytop)<contact_band))&xhit)
            for edge in (body[:,1]-body[:,0],body[:,3]-body[:,0]):
                axis=np.stack((-edge[:,1],edge[:,0]),axis=1)
                bounds=body[:,:,0]*axis[:,0,None]+body[:,:,1]*axis[:,1,None]
                middle=centers[None,:,0]*axis[:,0,None]+centers[None,:,1]*axis[:,1,None]
                radius=radii[None,:,0]*np.abs(axis[:,0,None])+radii[None,:,1]*np.abs(axis[:,1,None])
                lower=middle+radius-bounds.min(axis=1)[:,None]+tolerance
                upper=bounds.max(axis=1)[:,None]-(middle-radius)+tolerance
                sensitive|=((np.abs(lower)<contact_band)|(np.abs(upper)<contact_band))&box
                overlap&=(lower>=0)&(upper>=0)
            result[indices]=overlap.any(axis=1)
            # This preserves scalar floating-point contact decisions instead of
            # loosening geometric constraints to obtain speed.
            for local in np.flatnonzero(sensitive.any(axis=1)):
                result[indices[local]]=self.grid.intersects(body[local])
                self.last_scalar_fallbacks+=1
        return result
