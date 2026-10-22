"""Self-contained A* without the missing external Search_2D dependency."""

from navstack.planning import grid_search, GridAStarPlanner


class AStar:
    def __init__(
        self, s_start, s_goal, heuristic_type="octile", obstacles=(), grid_size=64
    ):
        if heuristic_type not in ("octile", "euclidean"):
            raise ValueError(
                "Use octile/euclidean for 8-neighbor sqrt(2) diagonal costs"
            )
        self.start, self.goal, self.obstacles, self.grid_size = (
            tuple(s_start),
            tuple(s_goal),
            set(obstacles),
            grid_size,
        )

    def searching(self):
        path, _, expanded = grid_search(
            self.start, self.goal, self.obstacles, self.grid_size
        )
        return path, {"expanded_nodes": expanded}
