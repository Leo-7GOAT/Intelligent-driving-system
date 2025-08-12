import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

# === 核心MPC参数与类 ===
class P:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 6  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 3.6 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 1.0  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7 * W  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= P.steer_max:
            return P.steer_max
        if delta <= -P.steer_max:
            return -P.steer_max
        return delta

    @staticmethod
    def limit_speed(v):
        if v >= P.speed_max:
            return P.speed_max
        if v <= P.speed_min:
            return P.speed_min
        return v


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        N = self.length
        window = P.N_IND
        search_inds = [(self.ind_old + i) % N for i in range(window)]
        dx = [node.x - x for x in self.cx[self.ind_old: (self.ind_old + P.N_IND)]]
        dy = [node.y - y for y in self.cy[self.ind_old: (self.ind_old + P.N_IND)]]
        dist = np.hypot(dx, dy)
        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind
        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])
        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])
        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]
        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    z_ref = np.zeros((P.NX, P.T + 1))
    length = ref_path.length
    ind, _ = ref_path.nearest_index(node)
    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = sp[ind]
    z_ref[3, 0] = ref_path.cyaw[ind]
    dist_move = 0.0
    for i in range(1, P.T + 1):
        dist_move += abs(node.v) * P.dt
        ind_move = int(round(dist_move / P.d_dist))
        index = min(ind + ind_move, length - 1)
        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = sp[index]
        z_ref[3, i] = ref_path.cyaw[index]
    return z_ref, ind


def linear_mpc_control(z_ref, z0, a_old, delta_old):
    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T
    x, y, yaw, v = None, None, None, None
    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)
        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])
        if max(du_a_max, du_d_max) < P.du_res:
            break
    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(z0, a, delta, z_ref):
    z_bar = z_ref * 0.0
    for i in range(P.NX):
        z_bar[i, 0] = z0[i]
    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])
    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        node.update(ai, di, 1.0)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw
    return z_bar


def calc_linear_discrete_model(v, phi, delta):
    A = np.array([[1.0, 0.0, P.dt * math.cos(phi), - P.dt * v * math.sin(phi)],
                  [0.0, 1.0, P.dt * math.sin(phi), P.dt * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])
    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [P.dt, 0.0],
                  [0.0, P.dt * v / (P.WB * math.cos(delta) ** 2)]])
    C = np.array([P.dt * v * math.sin(phi) * phi,
                  -P.dt * v * math.cos(phi) * phi,
                  0.0,
                  -P.dt * v * delta / (P.WB * math.cos(delta) ** 2)])
    return A, B, C


def solve_linear_mpc(z_ref, z_bar, z0, d_bar):
    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))
    cost = 0.0
    constrains = []
    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])
        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]
    cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)
    constrains += [z[:, 0] == z0]
    constrains += [z[2, :] <= P.speed_max]
    constrains += [z[2, :] >= P.speed_min]
    constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    try:
        prob.solve(solver=cvxpy.OSQP)
    except Exception as e:
        print(f"OSQP solve failed: {e}")
        return [0.0] * P.T, [0.0] * P.T, None, None, None, None
    a, delta, x, y, yaw, v = None, None, None, None, None, None
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")
        a = [0.0] * P.T
        delta = [0.0] * P.T
    return a, delta, x, y, yaw, v

def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        move_direction = math.atan2(dy, dx)
        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
    speed_profile[-1] = 0.0
    return speed_profile

def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi
    return angle

# ========== 主控函数：完整跟踪PPO输出参考轨迹 ==========
def mpc_track_ppo_path(cx, cy, cyaw=None, target_speed=2.0, plot=True, grid_size=4):
    """
    输入PPO轨迹点(cx, cy)，自动MPC全程跟踪并可视化
    :param cx: PPO参考x点序列
    :param cy: PPO参考y点序列
    :param cyaw: 参考yaw，若无则自动用差分估算
    :param target_speed: 目标速度（可调）
    :param plot: 是否画图
    :param grid_size: 地图尺寸（用于结果裁剪）
    :return: mpc_x, mpc_y
    """
    if cyaw is None:
        # 差分法自动估算yaw
        cyaw = np.zeros_like(cx)
        for i in range(1, len(cx)):
            dx = cx[i] - cx[i-1]
            dy = cy[i] - cy[i-1]
            cyaw[i] = math.atan2(dy, dx)
        cyaw[0] = cyaw[1]
    ck = np.zeros_like(cx)
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    mpc_ref_path = PATH(cx, cy, cyaw, ck)
    node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    a_opt, delta_opt = None, None
    mpc_x, mpc_y = [node.x], [node.y]
    steps = 0
    max_steps = len(cx) + 50
    while True:
        z_ref, target_ind = calc_ref_trajectory_in_T_step(node, mpc_ref_path, sp)
        z0 = [node.x, node.y, node.v, node.yaw]
        a_opt, delta_opt, *_ = linear_mpc_control(z_ref, z0, a_opt, delta_opt)
        node.update(a_opt[0], delta_opt[0], 1.0)
        mpc_x.append(node.x)
        mpc_y.append(node.y)
        steps += 1
        dx = node.x - cx[-1]
        dy = node.y - cy[-1]
        # 更严格的收敛条件：距离终点<0.2且速度<0.1
        if np.hypot(dx, dy) < 0.2 and abs(node.v) < 0.1:
            break
        if steps > max_steps:
            break
    # 可选：最后一帧对齐终点
    if np.hypot(mpc_x[-1] - cx[-1], mpc_y[-1] - cy[-1]) < 0.5:
        mpc_x[-1], mpc_y[-1] = cx[-1], cy[-1]
    # 可选：裁剪所有轨迹点到地图范围内
    mpc_x = np.clip(mpc_x, 0, grid_size-1)
    mpc_y = np.clip(mpc_y, 0, grid_size-1)
    if plot:
        plt.plot(cx, cy, '-o', label='参考路径')
        plt.plot(mpc_x, mpc_y, '-x', label='MPC跟踪轨迹')
        plt.legend(); plt.axis('equal'); plt.title("参考路径与MPC轨迹对比")
        plt.xlabel('x'); plt.ylabel('y'); plt.show()
    return np.array(mpc_x), np.array(mpc_y)

# 用法示例（注释掉，按需调用）：
# cx, cy = ... # 这里传入你的PPO/A*输出轨迹点
# mpc_x, mpc_y = mpc_track_ppo_path(cx, cy)
