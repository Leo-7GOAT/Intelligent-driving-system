# /*********************************************************************
#  * Software License Agreement (BSD License)
#  *
#  *  Copyright (c) 2010, Rice University
#  *  All rights reserved.
#  *
#  *  Redistribution and use in source and binary forms, with or without
#  *  modification, are permitted provided that the following conditions
#  *  are met:
#  *
#  *   * Redistributions of source code must retain the above copyright
#  *     notice, this list of conditions and the following disclaimer.
#  *   * Redistributions in binary form must reproduce the above
#  *     copyright notice, this list of conditions and the following
#  *     disclaimer in the documentation and/or other materials provided
#  *     with the distribution.
#  *   * Neither the name of the Rice University nor the names of its
#  *     contributors may be used to endorse or promote products derived
#  *     from this software without specific prior written permission.
#  *
#  *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  *  POSSIBILITY OF SUCH DAMAGE.
#  *********************************************************************/

"""Obstacle-free forward bounded-curvature lengths, not executable trajectories.

Six-family formulas adapted from OMPL DubinsStateSpace.cpp (Mark Moll).
Copyright (c) 2010, Rice University. BSD-3-Clause; full notice is retained in
third_party/ompl/LICENSE. No OMPL runtime, classifier or path sampler is bundled.
https://github.com/ompl/ompl/blob/main/src/ompl/base/spaces/src/DubinsStateSpace.cpp
"""
import math


def _angle(value):
    wrapped=value%(2*math.pi)
    return 0. if min(wrapped,2*math.pi-wrapped)<1e-10 else wrapped


def dubins_families(start,goal,radius):
    """Return feasible (word, three nonnegative segment lengths in metres)."""
    if (not math.isfinite(radius) or radius<=0 or len(start)!=3 or len(goal)!=3
            or not all(math.isfinite(v) for v in (*start,*goal))):
        raise ValueError('Finite poses and positive turning radius required')
    dx,dy=goal[0]-start[0],goal[1]-start[1]
    direction=math.atan2(dy,dx);d=math.hypot(dx,dy)/radius
    if not math.isfinite(d):raise ValueError('Normalized pose separation overflow')
    a,b=_angle(start[2]-direction),_angle(goal[2]-direction)
    if d<1e-12 and abs(math.atan2(math.sin(a-b),math.cos(a-b)))<1e-12:
        return [('LSL',(0.,0.,0.))]
    sa,ca,sb,cb=math.sin(a),math.cos(a),math.sin(b),math.cos(b)
    cosine=ca*cb+sa*sb;result=[]
    def add(word,t,p,q):result.append((word,(radius*_angle(t),radius*p,radius*_angle(q))))
    square=2+d*d-2*cosine+2*d*(sa-sb)
    if square>=-1e-10:
        turn=math.atan2(cb-ca,d+sa-sb)
        add('LSL',-a+turn,math.sqrt(max(0.,square)),b-turn)
    square=2+d*d-2*cosine+2*d*(sb-sa)
    if square>=-1e-10:
        turn=math.atan2(ca-cb,d-sa+sb)
        add('RSR',a-turn,math.sqrt(max(0.,square)),-b+turn)
    square=d*d-2+2*cosine-2*d*(sa+sb)
    if square>=-1e-10:
        straight=math.sqrt(max(0.,square))
        turn=math.atan2(ca+cb,d-sa-sb)-math.atan2(2.,straight)
        add('RSL',a-turn,straight,b-turn)
    square=d*d-2+2*cosine+2*d*(sa+sb)
    if square>=-1e-10:
        straight=math.sqrt(max(0.,square))
        turn=math.atan2(-ca-cb,d+sa+sb)-math.atan2(-2.,straight)
        add('LSR',-a+turn,straight,-b+turn)
    for word,value,turn,first_sign in (
        ('RLR',(6-d*d+2*cosine+2*d*(sa-sb))/8,math.atan2(ca-cb,d-sa+sb),1),
        ('LRL',(6-d*d+2*cosine-2*d*(sa-sb))/8,math.atan2(cb-ca,d+sa-sb),-1)):
        if abs(value)<=1+1e-10:
            middle=2*math.pi-math.acos(max(-1.,min(1.,value)))
            first=_angle(first_sign*(a-turn)+middle/2)
            last=_angle(first_sign*(a-b)-first+middle)
            result.append((word,(radius*first,radius*middle,radius*last)))
    return result


def dubins_distance(start,goal,radius):
    return min(sum(lengths) for _,lengths in dubins_families(start,goal,radius))
