"""8-neighbor A* with Euclidean costs and no corner cutting."""

from navstack.planning import grid_search


def plan_with_astar(start, goal, obstacles, grid_size):
    return grid_search(tuple(start), tuple(goal), set(obstacles), grid_size)[0]
