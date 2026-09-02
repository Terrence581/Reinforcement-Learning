import numpy as np


# Part1. 设置网格 与 插值处理

class BeliefGrid:
    
    def __init__(self, grid_n=401):
        self.grid_n = int(grid_n)
        self.grid = np.linspace(0.0, 1.0, self.grid_n)

    def interp(self, values, x):
        #  在 [0,1] 等距网格上的线性插值
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        pos = x * (self.grid_n - 1)
        idx = np.clip(pos.astype(int), 0, self.grid_n - 2)
        frac = pos - idx
        return (1.0 - frac) * values[idx] + frac * values[idx + 1]


def T_of_omega(omega, p01, p11):
    #  未观测时的信念推进：T(ω) = ω p11 + (1-ω) p01
    return omega * p11 + (1 - omega) * p01



# Part2. first crossing time

def first_crossing_time(start_w, w_threshold, p01, p11, max_iter=10000, return_path=True):
    """
    Function:
    从起始信念 start_w 出发，迭代 T(·)，找到第一个使 T^t(start_w) > w_threshold 的 t
    这就是“首次越界时间（first crossing time）”

    Input:
    - start_w      : 起始信念（如 p01 或 ω(1)）
    - w_threshold  : 阈值 ω_bar（论文灰色区构造中的 ω_bar}）
    - p01, p11     : 二态马氏转移参数
    - max_iter     : 安全上限，防止极端情况下无限循环
    - return_path  : 是否返回迭代轨迹（[(step,value), ...] 形式；默认保存但不打印）

    Return:
    - t_cross      : 首次越界的步数 t（若起点已 > 阈值，则返回 0）
    - w_cross      : 越界时对应的数值 T^t(start_w)
    - path         : 轨迹列表（[(0, start_w), (1, T(start_w)), ... , (t, w_cross)]）
    """
    t = 0
    w = float(start_w)
    path = [(0, w)] if return_path else None

    if w > w_threshold:
        return 0, w, path

    while t < max_iter and w <= w_threshold:
        w = T_of_omega(w, p01, p11)
        t += 1
        if return_path:
            path.append((t, w))
        if w > w_threshold:
            break

    return t, w, path



# Part3. 单臂情况：值迭代求 V -> 从 V 得策略 -> 在策略下评估 D

def value_iteration_single_arm(beta, m, p01, p11, grid: BeliefGrid, tol=1e-9, max_iter=10000):
    """
    求解在补贴 m 下的单臂 Bellman：
      V(ω;0) = m + β V(T(ω))
      V(ω;1) = ω + β [ ω V(p11) + (1-ω) V(p01) ]
      V(ω)   = max{ V(ω;0), V(ω;1) }

    Return:
      V: 网格上的值函数
      PI: 网格上的最优动作（0=被动, 1=主动）
      Q0: 被动时的价值
      Q1: 激活时的价值
    """
    V = np.zeros(grid.grid_n, dtype=float)
    idx_p11 = int(round(p11 * (grid.grid_n - 1)))
    idx_p01 = int(round(p01 * (grid.grid_n - 1)))

    for _ in range(max_iter):
        Tgrid = T_of_omega(grid.grid, p01, p11)
        V_T = grid.interp(V, Tgrid)

        # 主动动作的“下一步期望价值”，只依赖 V(p11) 与 V(p01)
        V_active_next = grid.grid * V[idx_p11] + (1 - grid.grid) * V[idx_p01]

        Q0 = m + beta * V_T
        Q1 = grid.grid + beta * V_active_next
        V_new = np.maximum(Q0, Q1)

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    PI = (Q1 >= Q0).astype(int)
    return V, PI, Q0, Q1


def discounted_passive_time_under_policy(beta, p01, p11, PI, grid: BeliefGrid, tol=1e-9, max_iter=10000):
    """
    在给定最优策略 PI 下评估“被动贴现总时长” D：
      若 PI(ω)=0: D(ω)=1+β D(T(ω))
      若 PI(ω)=1: D(ω)=β [ ω D(p11) + (1-ω) D(p01) ]
    """
    D = np.zeros(grid.grid_n, dtype=float)
    idx_p11 = int(round(p11 * (grid.grid_n - 1)))
    idx_p01 = int(round(p01 * (grid.grid_n - 1)))

    for _ in range(max_iter):
        Tgrid = T_of_omega(grid.grid, p01, p11)
        D_T = grid.interp(D, Tgrid)
        D_p11 = D[idx_p11]
        D_p01 = D[idx_p01]
        D_new = np.where(PI == 0,
                         1.0 + beta * D_T,
                         beta * (grid.grid * D_p11 + (1 - grid.grid) * D_p01))
        if np.max(np.abs(D_new - D)) < tol:
            D = D_new
            break
        D = D_new
    return D



# Part4. 单臂情况：计算 Whittle Index 即 Wβ(ω)

def whittle_index_numeric(beta, p01, p11, omega, grid: BeliefGrid, mL=0, mU=1.0, expand=8, iters=22):
    """
    计算 Wβ(ω)：最小的 m 使得“被动”与“主动”在 ω 处无差别
    做法：对 m 二分。每次给定 m, 解单臂 Bellman, 直接比较 Q1(ω)-Q0(ω) 的符号

    Return:
      Wβ(ω) 的近似值
    """
    #  把 ω 对齐到网格
    idx_w = int(round(float(np.clip(omega, 0.0, 1.0)) * (grid.grid_n - 1)))

    def A(m):
        V, PI, Q0, Q1 = value_iteration_single_arm(beta, m, p01, p11, grid)
        return Q1[idx_w] - Q0[idx_w]

    aL, aU = A(mL), A(mU)
    # 扩展区间直到 A(mL) >= 0 >= A(mU)
    exp = 0
    while aL < 0 and exp < expand:
        mL -= 1.0
        aL = A(mL)
        exp += 1
    exp = 0
    while aU > 0 and exp < expand:
        mU += 1.0
        aU = A(mU)
        exp += 1

    # 二分
    for _ in range(iters):
        mM = 0.5 * (mL + mU)
        aM = A(mM)
        if aM >= 0:
            mL = mM
        else:
            mU = mM
    return 0.5 * (mL + mU)



# Part5. 多臂情况：Fig.12 的上界算法

def compute_upper_bound_eps(beta, K, p01_list, p11_list, omega1_list, eps, grid: BeliefGrid):
    """
    Input:
      - beta, K
      - 每条臂的 p01, p11
      - 初始信念 ω_i(1)
      - eps: 允许的上界误差 ε
      - grid: 网格

    Output:
      - m_prime      : 算法定位的补贴（若区间非灰且 D≥0，则等于 m*；若在灰区误差 ≤ ε(1-β)/K）
      - G_upper      : 对应的性能上界 G_{β,m}(Ω(1))
      - stop_interval: 停靠的小区间 [a_j, a_{j+1}]
      - gray_ranges  : 灰色区间列表（并集 V 的构成）
    """
    N = len(p01_list)
    p01_arr = np.asarray(p01_list, dtype=float)
    p11_arr = np.asarray(p11_list, dtype=float)
    w1_arr  = np.asarray(omega1_list, dtype=float)

    delta = eps * (1 - beta) / K

    # ---------- Step 1: 负相关臂 ----------
    candidates = []     # 存放用于分段的所有 Whittle 指数取值
    neg_idx = np.where(p11_arr < p01_arr)[0]
    for i in neg_idx:
        # 基本三点：p11, p01, T(p11)
        for x in (p11_arr[i], p01_arr[i], T_of_omega(p11_arr[i], p01_arr[i], p11_arr[i])):
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i], x, grid))
        # 初值相关点
        w_o = p01_arr[i] / (p01_arr[i] + (1 - p11_arr[i]))
        if w1_arr[i] < w_o:
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w1_arr[i], grid))
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i],
                                                    T_of_omega(w1_arr[i], p01_arr[i], p11_arr[i]), grid))
        else:
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w1_arr[i], grid))

    # ---------- Step 2: 正相关臂 + 灰色区 ----------
    gray_ranges = []    # 灰色区的若干小区间
    pos_idx = np.where(p11_arr >= p01_arr)[0]
    for i in pos_idx:
        # 基本三点：p01, p11, 稳态 ω_o
        w_from_p01 = whittle_index_numeric(beta, p01_arr[i], p11_arr[i], p01_arr[i], grid)
        w_from_p11 = whittle_index_numeric(beta, p01_arr[i], p11_arr[i], p11_arr[i], grid)
        w_o = p01_arr[i] / (p01_arr[i] + (1 - p11_arr[i]))
        w_from_omg0 = whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w_o, grid)
        candidates.extend([w_from_p01, w_from_p11, w_from_omg0])

        # 在 [ω_o - δ/N, ω_o) 搜一个 ω_bar，满足 Wβ(ω_o) ≥ Wβ(ω_bar) - δ/N
        w_threshold = max(0.0, w_o - delta / N)
        # 用 first_crossing_time 来替代隐式 while：计算 l_i（从 p01 出发）
        t_cross_from_p01, w_cross_p01, p01_path = first_crossing_time(
            start_w=p01_arr[i], w_threshold=w_threshold, p01=p01_arr[i], p11=p11_arr[i],
            max_iter=10000, return_path=True
        )
        # 收集 Wβ(T^k(p01))，k=1..l_i
        w_tmp = p01_arr[i]
        for _ in range(1, t_cross_from_p01 + 1):
            w_tmp = T_of_omega(w_tmp, p01_arr[i], p11_arr[i])
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w_tmp, grid))

        # 若 ω_i(1) < ω_o，同样计算 d_i（从初始信念出发）
        min_right_val = w_from_omg0
        if w1_arr[i] < w_o:
            t_cross_from_init, w_cross_init, init_path = first_crossing_time(
                start_w=w1_arr[i], w_threshold=w_threshold, p01=p01_arr[i], p11=p11_arr[i],
                max_iter=10000, return_path=True
            )
            w_tmp = w1_arr[i]
            w_list = []
            for _ in range(1, t_cross_from_init + 1):
                w_tmp = T_of_omega(w_tmp, p01_arr[i], p11_arr[i])
                wk = whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w_tmp, grid)
                w_list.append(wk)
                candidates.append(wk)
            if len(w_list) > 0:
                min_right_val = min(min_right_val, w_list[-1])
        else:
            candidates.append(whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w1_arr[i], grid))

        # 灰色区间端点（左端取两者较小，右端为 Wβ(ω_o)）
        left_end = w_from_omg0
        if t_cross_from_p01 > 0:
            # 计算最后一次 T^l(p01) 的 Wβ
            w_last = p01_arr[i]
            for _ in range(1, t_cross_from_p01 + 1):
                w_last = T_of_omega(w_last, p01_arr[i], p11_arr[i])
            left_end = min(left_end, whittle_index_numeric(beta, p01_arr[i], p11_arr[i], w_last, grid))
        left_end = min(left_end, min_right_val)
        gray_ranges.append((left_end, w_from_omg0))

    # ---------- Step 3: 排序所有指数，形成分段端点 a ----------
    breakpoints = sorted(set([0.0, 1.0] + candidates))
    # a0=0, a_{h+1}=1 已包含在 breakpoints 中

    # ---------- Step 4: 非灰区间内检查 D 是否 ≥ 0 ----------
    def in_gray(m_val):
        # 判断一个 m 是否落入灰色区 V
        for Lg, Rg in gray_ranges:
            if Lg <= m_val <= Rg:
                return True
        return False

    def total_D_at_m(m_val):
        # 计算 ∑_i D_{β,m}(ω_i(1))，用于区间判定
        total = 0.0
        for i in range(N):
            V, PI, *_ = value_iteration_single_arm(beta, m_val, p01_arr[i], p11_arr[i], grid)
            D = discounted_passive_time_under_policy(beta, p01_arr[i], p11_arr[i], PI, grid)
            total += float(grid.interp(D, w1_arr[i]))
        return total

    target = (N - K) / (1 - beta)

    stop_interval = None
    for j in range(len(breakpoints) - 1):
        L, R = breakpoints[j], breakpoints[j + 1]
        mid = 0.5 * (L + R)
        if in_gray(mid):
            # 灰区：跳过
            continue
        # 该区间内最优策略不变 ⇒ ∑D 常数；用中点 mid 评估即可
        Dsum = total_D_at_m(mid)
        if Dsum - target >= 0:
            stop_interval = (L, R)
            break

    if stop_interval is None:
        # 理论上很少出现；稳妥退回最后一个区间
        stop_interval = (breakpoints[-2], breakpoints[-1])

    L, R = stop_interval
    m_prime = L  # Fig.12 输出区间左端 a_j 作为 m'（非灰区时 m'=m*）

    # ---------- Step 5: 在该区间计算上界 G ----------
    # 取区间中点（区间内策略不变，上界仅差线性项，任意取值等价）
    m_use = 0.5 * (L + R)
    Vsum = 0.0
    for i in range(N):
        V, PI, *_ = value_iteration_single_arm(beta, m_use, p01_arr[i], p11_arr[i], grid)
        Vsum += float(grid.interp(V, w1_arr[i]))
    G_upper = Vsum - m_use * target

    return m_prime, G_upper, stop_interval, gray_ranges

