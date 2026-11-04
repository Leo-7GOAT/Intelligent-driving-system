"""Bounded interior task-goal poses;no change to the task's success region."""
import math


def candidate_poses(goal,position_tolerance,yaw_tolerance):
    if (len(goal)!=3 or not all(math.isfinite(x) for x in goal)
            or not math.isfinite(position_tolerance) or position_tolerance<=0
            or not math.isfinite(yaw_tolerance) or not 0<yaw_tolerance<=math.pi):
        raise ValueError('Finite goal and positive finite task tolerances required')
    x,y,yaw=map(float,goal);co,si=math.cos(yaw),math.sin(yaw)
    # Each noncentral coordinate is half the task radius: diagonal positions
    # remain at radius/sqrt(2),leaving tracking room inside the unchanged disk.
    d=position_tolerance/2;a=yaw_tolerance/2
    offsets=[(u,v,w) for u in (0.,-d,d) for v in (0.,-d,d) for w in (0.,-a,a)]
    offsets.sort(key=lambda q:(q!=(0.,0.,0.),q[0]**2+q[1]**2,abs(q[2]),q))
    poses=[(x+u*co-v*si,y+u*si+v*co,(yaw+w+math.pi)%(2*math.pi)-math.pi) for u,v,w in offsets]
    poses[0]=(x,y,yaw)  # Preserve the exact original libm input,including+pi.
    return poses


def connect_region(start,goal,world,vehicle,position_tolerance,yaw_tolerance,connector):
    """Keep a feasible exact connection;otherwise try26interior poses.

    The injected connector performs its usual full-body checks. A result is
    geometric evidence,not a closed-loop tracking or dynamic-safety guarantee.
    """
    poses=candidate_poses(goal,position_tolerance,yaw_tolerance);best=None;queries=0
    for index,pose in enumerate(poses):
        path=connector(start,pose,world,vehicle);queries+=1
        if path is None:continue
        if index==0:return {'target':pose,'path':path,'queries':queries,'exact_goal':True}
        length=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:]))
        key=(length,index)
        if best is None or key<best[0]:best=(key,pose,path)
    return None if best is None else {'target':best[1],'path':best[2],'queries':queries,'exact_goal':False}
