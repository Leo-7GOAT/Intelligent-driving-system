"""Read-only local lane projection diagnostics; Python 3.7 compatible.

The finite lane search is not a complete road mesh. It is used only by the
explicitly versioned v2/v3 native-road geometry, never legacy runs.
"""
import math


def refined_lane_point(road_map, waypoint, location):
    """Minimize planar centerline distance within +/-1m of the reported s.

    A deterministic coarse bracket followed by24golden-section refinements
    avoids assuming that lane direction agrees with increasing OpenDRIVE s.
    No lane-width dilation or position change is applied.
    """
    samples = {}
    def sample(s):
        if s not in samples:
            w = road_map.get_waypoint_xodr(waypoint.road_id,waypoint.lane_id,float(s))
            if w is None or w.section_id != waypoint.section_id:
                samples[s] = (math.inf,None)
            else:
                p = w.transform.location
                samples[s] = ((p.x-location.x)**2+(p.y-location.y)**2,w)
        return samples[s]
    grid = [waypoint.s-1+i*.1 for i in range(21)]
    best = min(range(len(grid)),key=lambda i:sample(grid[i])[0])
    left,right = grid[max(0,best-1)],grid[min(20,best+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(24):
        a,b = right-ratio*(right-left),left+ratio*(right-left)
        if sample(a)[0] <= sample(b)[0]: right=b
        else: left=a
    sample((left+right)/2)
    distance2,w = min(samples.values(),key=lambda item:item[0])
    if w is None:
        return None
    p = w.transform.location
    return {"road_id":w.road_id,"lane_id":w.lane_id,"section_id":w.section_id,"s":w.s,
            "center":[p.x,p.y,p.z],"distance":math.sqrt(distance2),"half_width":w.lane_width/2,
            "inside":distance2<(w.lane_width/2)**2,"height_error":abs(p.z-location.z),
            "queries":len(samples)}


def local_driving_union(road_map, location, driving_type):
    first = road_map.get_waypoint(location,project_to_road=True,lane_type=driving_type)
    if first is None:
        return {"inside":False,"candidates":[]}
    todo,seen,candidates = [first],set(),[]
    while todo and len(seen)<16:
        w = todo.pop(0)
        identity = (w.road_id,w.section_id,w.lane_id)
        if identity in seen or w.lane_type != driving_type:
            continue
        seen.add(identity)
        row = refined_lane_point(road_map,w,location)
        if row is not None:
            candidates.append(row)
        for neighbour in (w.get_left_lane(),w.get_right_lane()):
            if neighbour is not None and neighbour.lane_type == driving_type:
                todo.append(neighbour)
    return {"inside":any(r["inside"] and r["height_error"]<1.5 for r in candidates),
            "candidates":candidates}


def on_driving_surface(road_map, location, driving_type):
    """Fast raw-query acceptance, then refined union without width dilation."""
    raw = road_map.get_waypoint(location,project_to_road=False,lane_type=driving_type)
    if raw is not None and abs(raw.transform.location.z-location.z)<1.5:
        return True
    return local_driving_union(road_map,location,driving_type)['inside']


class DrivingSurfaceIndex:
    """Index nearby Driving lanes across road IDs, including junction links.

    One-metre OpenDRIVE samples discover candidates; refined lane widths decide
    membership. The lookup radius is candidate discovery, not road dilation.
    It remains a map-lane approximation, not a rendered asphalt-mesh proof.
    """
    def __init__(self,road_map,driving_type):
        self.road_map,self.driving_type=road_map,driving_type
        self.buckets={};self.max_half_width=0.
        for w in road_map.generate_waypoints(1.):
            if w.lane_type != driving_type:continue
            p=w.transform.location
            self.buckets.setdefault((math.floor(p.x/5),math.floor(p.y/5)),[]).append((p,w))
            self.max_half_width=max(self.max_half_width,w.lane_width/2)

    def inspect(self,location):
        radius=self.max_half_width+2.
        nearest={}
        for x in range(math.floor((location.x-radius)/5),math.floor((location.x+radius)/5)+1):
            for y in range(math.floor((location.y-radius)/5),math.floor((location.y+radius)/5)+1):
                for p,w in self.buckets.get((x,y),[]):
                    distance=math.hypot(p.x-location.x,p.y-location.y)
                    key=(w.road_id,w.section_id,w.lane_id)
                    if distance<=radius and abs(p.z-location.z)<1.5 and (key not in nearest or distance<nearest[key][0]):
                        nearest[key]=(distance,w)
        rows=[refined_lane_point(self.road_map,w,location) for _,w in nearest.values()]
        rows=[r for r in rows if r is not None]
        return {'inside':any(r['inside'] and r['height_error']<1.5 for r in rows),'candidates':rows}

    def contains(self,location):
        raw=self.road_map.get_waypoint(location,project_to_road=False,lane_type=self.driving_type)
        if raw is not None and abs(raw.transform.location.z-location.z)<1.5:return True
        return self.inspect(location)['inside']
