# astar_agent.py

import heapq

def heuristic(a, b):
    """曼哈顿距离启发式"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, grid_size, obstacles):
    # 八邻域（包含对角线移动）
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
            neighbors.append((nx, ny))
    return neighbors

def plan_with_astar(start, goal, obstacles, grid_size):
    """
    A*路径规划主接口
    :param start: (x, y) tuple, 起点
    :param goal: (x, y) tuple, 终点
    :param obstacles: set of (x, y), 障碍物集合
    :param grid_size: int, 地图尺寸
    :return: [ (x0, y0), (x1, y1), ... ]  路径点序列（包含起点终点），若无解则返回[]
    """
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # 回溯得到路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, grid_size, obstacles):
            tentative_g = g_score[current] + 1  # 所有邻居步长为1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))
    # 找不到路径
    return []

# 可选：模块自测
if __name__ == "__main__":
    grid_size = 4
    obstacles = {(1,1), (2,2), (3,3)}
    start = (0, 0)
    goal = (4, 4)
    path = plan_with_astar(start, goal, obstacles, grid_size)
    print("A*路径：", path)
