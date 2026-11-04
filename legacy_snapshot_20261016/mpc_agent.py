# mpc_agent.py

import numpy as np
from mpc_module import mpc_track_ppo_path
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ======== Curve smoothing interpolation function integrated here ========
def smooth_path(path, num=50, tail_extension=5):
    """
    Smooth and interpolate discrete path points (e.g., from A*) using B-spline,
    and add a tail extension in the direction of the last segment.
    :param path: List of (x, y) tuples
    :param num: Number of interpolated points
    :param tail_extension: Number of extra points to extend beyond the endpoint
    :return: List of (x, y) tuples
    """
    if len(path) < 4:
        return path
    x = [float(p[0]) for p in path]
    y = [float(p[1]) for p in path]
    # perform B-spline interpolation
    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=num)
    out = splev(unew, tck)
    smooth_xy = list(zip(out[0], out[1]))

    # ====== Tail extension: extend beyond endpoint ======
    if tail_extension > 0:
        x1, y1 = smooth_xy[-2]
        x2, y2 = smooth_xy[-1]
        dx, dy = x2 - x1, y2 - y1
        for i in range(1, tail_extension + 1):
            fx = x2 + dx * i
            fy = y2 + dy * i
            smooth_xy.append((fx, fy))
    return smooth_xy

# ======== Main interface ========
def mpc_track_path(path_xy, target_speed=2.0, grid_size=4, smooth=True):
    """
    MPC trajectory tracking interface.
    :param path_xy: List of reference (x, y) path points (e.g., from A* or PPO)
    :param target_speed: Desired tracking speed (m/s)
    :param grid_size: Size of the map (default 4)
    :param smooth: Whether to apply smoothing interpolation
    :return: List of (x, y) points representing the tracked trajectory
    """
    if not path_xy or len(path_xy) < 2:
        print("[MPC_AGENT] Input path too short for MPC tracking.")
        return []

    # ====== Smooth the reference path ======
    if smooth:
        path_xy = smooth_path(path_xy, num=50)

    cx = np.array([float(x) for x, y in path_xy])
    cy = np.array([float(y) for x, y in path_xy])

    mpc_x, mpc_y = mpc_track_ppo_path(
        cx, cy, target_speed=target_speed, plot=False, grid_size=grid_size
    )
    mpc_path = list(zip(mpc_x, mpc_y))

    # Calculate RMSE if reference and tracked path lengths match
    if path_xy and len(path_xy) == len(mpc_path):
        err = np.hypot(
            np.array([p[0] for p in mpc_path]) - np.array([p[0] for p in path_xy]),
            np.array([p[1] for p in mpc_path]) - np.array([p[1] for p in path_xy])
        )
        rmse = np.sqrt((err ** 2).mean())
        print(f"[METRIC] MPC_RMSE = {rmse:.3f} m")
        with open("metrics.csv", "a") as f:
            f.write(f"mpc_rmse,{rmse:.4f}\n")
    return mpc_path


def plot_path_compare(a_star_path, smooth_xy, mpc_path, goal=None, title="Trajectory Comparison"):
    """
    Plot comparison of A* path, smoothed path, and MPC tracked path.
    :param a_star_path: List of discrete (x, y) points from A*
    :param smooth_xy: List of smoothed/interpolated (x, y) points
    :param mpc_path: List of MPC-tracked (x, y) points
    :param goal: Optional goal point for marking on the plot
    :param title: Title of the plot
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # Plot A* discrete path
    ax.plot([p[0] for p in a_star_path], [p[1] for p in a_star_path], 'ks-', label='A* Path', zorder=1)
    # Plot smoothed path
    ax.plot([p[0] for p in smooth_xy], [p[1] for p in smooth_xy], 'g--', label='Smoothed Path', zorder=2)
    # Plot MPC tracked trajectory
    ax.plot([p[0] for p in mpc_path], [p[1] for p in mpc_path], 'bo-', label='MPC Tracked Path', zorder=3)

    # Start point
    ax.scatter(a_star_path[0][0], a_star_path[0][1], marker='o', label='Start', zorder=4)
    # Goal point
    end_pt = goal if goal is not None else a_star_path[-1]
    ax.scatter(end_pt[0], end_pt[1], marker='*', label='Goal', zorder=5)
    # MPC final point
    ax.scatter(mpc_path[-1][0], mpc_path[-1][1], marker='P', edgecolors='k', label='Agent Final', zorder=6)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    plt.show()

# Optional: module self-test
if __name__ == "__main__":
    # Example A* discrete path
    a_star_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 3), (4, 4)]
    # Perform MPC tracking (with smoothing)
    mpc_path = mpc_track_path(a_star_path)
    print("MPC tracked path:", mpc_path)
