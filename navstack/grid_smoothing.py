"""Optional B2 spline optimization after 2-D search, with swept-body certification.

Optimization is a local heuristic. Only the independent continuous-interval
certificate authorizes a returned path; optimizer convergence alone never does.
"""
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def certify_curve(curve, vehicle, world, spacing=.02):
    """Bound speed, curvature and body displacement over every cubic interval.

    For half-interval h, Taylor bounds on the first/second derivatives give a
    positive speed lower bound and an angular-speed upper bound. Inflating the
    midpoint body by h*(speed_upper + body_radius*angular_speed_upper) covers
    every body point throughout that interval, including rotation. Knot
    boundaries are always interval boundaries (the third derivative can jump).
    This certifies the spatial spline, not tracking errors or dynamic traffic.
    """
    left, right = [], []
    for a, b in zip(curve.x[:-1], curve.x[1:]):
        edges = np.linspace(a, b, max(2, math.ceil((b-a)/spacing)+1))
        left.extend(edges[:-1]); right.extend(edges[1:])
    left, right = np.asarray(left), np.asarray(right)
    middle, half = (left+right)/2, (right-left)/2
    first, second, third = curve(middle, 1), curve(middle, 2), curve(middle, 3)
    speed = np.linalg.norm(first, axis=1)
    acceleration = np.linalg.norm(second, axis=1)
    jerk = np.linalg.norm(third, axis=1)
    change = acceleration*half + .5*jerk*half**2
    lower, upper = speed-change, speed+change
    if np.any(lower <= .05):
        return None, {"reason":"stationary_or_uncertified_tangent"}
    cross = np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])
    curvature_bound = (cross + half*upper*jerk)/lower**3
    if np.max(curvature_bound) > 1/vehicle.min_turn_radius:
        return None, {"reason":"curvature_bound", "maximum":float(curvature_bound.max())}
    radius = math.hypot(max(vehicle.front, vehicle.rear), vehicle.width/2)
    margin = half*(upper + radius*(acceleration+jerk*half)/lower)
    middle_path = np.c_[curve(middle), np.arctan2(first[:,1], first[:,0])]
    # A common maximum is conservative and keeps the vectorized world API.
    free = world.free_many(middle_path, vehicle, float(margin.max()))
    if not free.all():
        return None, {"reason":"swept_body_collision", "intervals_rejected":int((~free).sum()),
                      "margin_m":float(margin.max())}
    samples = np.r_[left, right[-1]]
    derivative = curve(samples, 1)
    path = np.c_[curve(samples), np.unwrap(np.arctan2(derivative[:,1], derivative[:,0]))]
    return path, {"reason":None, "intervals":len(middle), "margin_m":float(margin.max()),
                  "curvature_upper_bound":float(curvature_bound.max())}


def body_spline(raw_xy, start, goal, world, vehicle):
    """Deterministic bounded postprocessor; no Hybrid A*, teacher or rule route.

    The 2-D A* route seeds endpoint-tangent cubic knots about four metres apart.
    A 150-evaluation least-squares budget penalizes full-body SAT overlap and
    curvature. The nearest 64 obstacle rectangles are optimization guidance;
    final certification checks ALL relevant geometry through World.free_many.
    """
    xy = np.asarray(raw_xy, float)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0),axis=1)>1e-7]
    xy = xy[keep]
    s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if len(s)<2 or s[-1]<.1:
        return None
    knots = np.linspace(0,s[-1],max(4,math.ceil(s[-1]/4)+1))
    points = np.c_[np.interp(knots,s,xy[:,0]),np.interp(knots,s,xy[:,1])]
    points[0], points[-1] = start[:2], goal[:2]
    sample = np.linspace(0,s[-1],math.ceil(s[-1]/.2)+1)
    tangents = ((1,[math.cos(start[2]),math.sin(start[2])]),
                (1,[math.cos(goal[2]),math.sin(goal[2])]))
    rects = list(world.static)
    if world.road is not None:
        road = world.road
        yy,xx = np.nonzero(~road.collision_drivable)
        rects.extend(np.c_[xx*road.dx,yy*road.dy,(xx+1)*road.dx,(yy+1)*road.dy].tolist())
    rects = np.asarray(rects,float).reshape(-1,4)
    centers, radii = (rects[:,:2]+rects[:,2:])/2, (rects[:,2:]-rects[:,:2])/2
    tree = cKDTree(centers) if len(rects) else None
    hl,hw = (vehicle.front+vehicle.rear)/2,vehicle.width/2

    def curve_for(z):
        return CubicSpline(knots,np.vstack([points[0],z.reshape(-1,2),points[-1]]),bc_type=tangents)

    def residual(z):
        curve = curve_for(z)
        pos,d1,d2 = curve(sample),curve(sample,1),curve(sample,2)
        norm = np.linalg.norm(d1,axis=1)
        k = (d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum(norm**3,1e-10)
        co,si = (d1/np.maximum(norm[:,None],1e-10)).T
        bodycenter = pos+np.c_[co,si]*((vehicle.front-vehicle.rear)/2)
        extent_x,extent_y = np.abs(co)*hl+np.abs(si)*hw,np.abs(si)*hl+np.abs(co)*hw
        boundary = np.minimum.reduce([bodycenter[:,0]-extent_x,world.width-bodycenter[:,0]-extent_x,
                                      bodycenter[:,1]-extent_y,world.height-bodycenter[:,1]-extent_y])
        clearance = boundary
        if tree is not None:
            _,ids = tree.query(bodycenter,k=list(range(1,min(64,len(rects))+1)))
            delta = centers[ids]-bodycenter[:,None,:]
            dx,dy = delta[:,:,0],delta[:,:,1]
            # Advanced indexing retains [pose, nearby_rectangle, coordinate].
            near_radii = radii[ids]; rx,ry = near_radii[:,:,0],near_radii[:,:,1]
            c,t = co[:,None],si[:,None]
            gaps = [np.abs(dx)-(extent_x[:,None]+rx),np.abs(dy)-(extent_y[:,None]+ry),
                    np.abs(dx*c+dy*t)-(hl+np.abs(c)*rx+np.abs(t)*ry),
                    np.abs(-dx*t+dy*c)-(hw+np.abs(t)*rx+np.abs(c)*ry)]
            clearance = np.minimum(boundary,np.maximum.reduce(gaps).min(axis=1))
        return np.r_[30*np.maximum(.10-clearance,0),40*np.maximum(np.abs(k)-.95/vehicle.min_turn_radius,0),
                     10*np.maximum(.2-norm,0),.02*(z-points[1:-1].ravel()),.05*d2.ravel()]

    fitted = least_squares(residual,points[1:-1].ravel(),max_nfev=150,ftol=1e-7,xtol=1e-7)
    curve = curve_for(fitted.x)
    # Refinement tightens a conservative bound; it never relaxes curvature or
    # geometry limits. A finite five-level budget also covers near-tight paths.
    for spacing in (.02,.01,.005,.0025,.00125):
        path,_ = certify_curve(curve,vehicle,world,spacing=spacing)
        if path is not None:
            return path
    return None
