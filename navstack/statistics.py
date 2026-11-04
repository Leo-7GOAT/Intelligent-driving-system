"""Seed-level paired uncertainty. Episodes are never counted as training replications."""
import numpy as np


def paired_bootstrap(left, right, repetitions=20000, seed=20261016):
    if repetitions<10000:raise ValueError("At least 10000 bootstrap repetitions required")
    shared=sorted(set(left)&set(right))
    a=np.array([left[k] for k in shared],float);b=np.array([right[k] for k in shared],float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Finite seed-level metrics required")
    delta=a-b
    result={"independent_training_seeds":shared,"n":len(shared),"per_seed_difference":dict(zip(shared,delta.tolist())),
            "left_mean":float(a.mean()) if len(a) else None,"right_mean":float(b.mean()) if len(b) else None,
            "left_std":float(a.std(ddof=1)) if len(a)>1 else None,"right_std":float(b.std(ddof=1)) if len(b)>1 else None,
            "difference_mean":float(delta.mean()) if len(delta) else None,"ci95":None,
            "replicates":repetitions,"statistics_seed":seed,"unit":"independent_training_seed",
            "unpaired_left_seeds":sorted(set(left)-set(right)),"unpaired_right_seeds":sorted(set(right)-set(left))}
    if len(shared)<3:
        result["interval_status"]="insufficient_independent_seeds_minimum_3";return result
    rng=np.random.default_rng(seed)
    estimates=delta[rng.integers(0,len(delta),(repetitions,len(delta)))].mean(axis=1)
    result["ci95"]=np.percentile(estimates,[2.5,97.5]).tolist()
    result["interval_status"]="computed_percentile_paired_seed_bootstrap"
    return result


def learning_curve_auc(points,lower=100000,upper=1000000):
    points=sorted(points)
    if not points or points[0][0]>lower or points[-1][0]<upper:
        return {"value":None,"status":"milestone_interval_incomplete","interval":[lower,upper]}
    x,y=np.array(points,float).T
    internal=(x>lower)&(x<upper)
    xx=np.r_[lower,x[internal],upper];yy=np.interp(xx,x,y)
    return {"value":float(np.trapezoid(yy,xx)/(upper-lower)),"interval":[lower,upper],"status":"computed"}
