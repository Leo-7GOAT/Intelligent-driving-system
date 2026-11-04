import numpy as np


def timing_summary(values, period):
    if not values:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "over_period_fraction": None,
        }
    a = np.asarray(values)
    return {
        "count": len(a),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(max(a)),
        "over_period_fraction": float(np.mean(a > period)),
        "period": period,
        "unit": "seconds",
    }
