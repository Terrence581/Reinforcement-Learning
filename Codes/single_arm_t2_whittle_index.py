import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SearchConfig:
    l_max: int = 350
    j0: int = 240
    j1: int = 120
    r_max: int = 8
    eps: float = 1e-8
    bisect_steps: int = 80


@dataclass
class WhittleResult:
    success: bool
    hat_w: Optional[float]
    residual: Optional[float]
    message: str
    interval: Tuple[float, float]
    rounds: int
    history: List[Dict[str, float]]


def _validate_inputs(omega: np.ndarray, p: np.ndarray, b: np.ndarray, beta: float) -> None:
    if p.shape != (3, 3):
        raise ValueError("P must be a 3x3 matrix.")
    if b.shape != (3,):
        raise ValueError("B must be a length-3 vector.")
    if omega.shape != (3,):
        raise ValueError("omega must be a length-3 vector.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must satisfy 0 < beta < 1.")
    if np.any(p < -1e-12):
        raise ValueError("P must be nonnegative.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Each row of P must sum to 1.")
    if not np.isclose(omega.sum(), 1.0, atol=1e-8) or np.any(omega < -1e-12):
        raise ValueError("omega must be a probability vector.")
    if not np.all(np.diff(b) >= -1e-12):
        raise ValueError("B must satisfy B0 <= B1 <= B2.")
    if not np.isclose(b[0], 0.0, atol=1e-8):
        raise ValueError("B0 must be 0.")


def _gamma_vector(p: np.ndarray, b: np.ndarray, m: float) -> np.ndarray:
    # [max{p0 B', m}, max{p1 B', m}, max{p2 B', m}]'
    return np.maximum(p @ b, m)


def _r2_value(
    z: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    beta: float,
    m: float,
    gamma: np.ndarray,
) -> float:
    # r2(z) = zB' + beta*z*Gamma(m) - m - beta*max{T1(z)B', m}
    z_next = z @ p
    return float(z @ b + beta * (z @ gamma) - m - beta * max(float(z_next @ b), m))


def _compute_l_and_state(
    omega_1: np.ndarray,
    omega: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    beta: float,
    m: float,
    l_max: int,
    gamma: np.ndarray,
    r2_omega: float,
) -> Tuple[float, Optional[np.ndarray]]:
    # L(omega1, omega) = min_k { r2(T^k(omega1)) > r2(omega) }.
    z_k = omega_1.copy()
    for k in range(l_max + 1):
        z_k_plus_1 = z_k @ p
        r2_zk = z_k @ b + beta * (z_k @ gamma) - m - beta * max(float(z_k_plus_1 @ b), m)
        if r2_zk > r2_omega:
            return float(k), z_k.copy()
        z_k = z_k_plus_1
    return math.inf, None


def _f_and_g_from_l(
    l_value: float,
    z_l: Optional[np.ndarray],
    beta: float,
) -> Tuple[float, np.ndarray]:
    # Follow the same handling as in the pseudocode:
    # if L = +inf, set beta^L = 0 and (1-beta^L)/(1-beta) = 1/(1-beta).
    if math.isinf(l_value):
        return 1.0 / (1.0 - beta), np.zeros(3)
    l_int = int(l_value)
    f_value = (1.0 - beta**l_int) / (1.0 - beta)
    g_value = (beta**l_int) * z_l
    return float(f_value), g_value


def compute_residual(
    omega: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    beta: float,
    m: float,
    l_max: int,
) -> Tuple[float, Dict[str, object]]:
    omega = np.asarray(omega, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3, 3)
    b = np.asarray(b, dtype=float).reshape(3)

    gamma = _gamma_vector(p, b, m)
    r2_omega = _r2_value(omega, p, b, beta, m, gamma)

    p0, p1, p2 = p[0], p[1], p[2]
    l_p0, z_l_p0 = _compute_l_and_state(p0, omega, p, b, beta, m, l_max, gamma, r2_omega)
    l_p1, z_l_p1 = _compute_l_and_state(p1, omega, p, b, beta, m, l_max, gamma, r2_omega)
    l_p2, z_l_p2 = _compute_l_and_state(p2, omega, p, b, beta, m, l_max, gamma, r2_omega)

    f_p0, g_p0 = _f_and_g_from_l(l_p0, z_l_p0, beta)
    f_p1, g_p1 = _f_and_g_from_l(l_p1, z_l_p1, beta)
    f_p2, g_p2 = _f_and_g_from_l(l_p2, z_l_p2, beta)

    # F(P), G(P), H(P) follow Theorem 3 notation.
    f_p = np.array([f_p0, f_p1, f_p2], dtype=float)
    g_p = np.vstack([g_p0, g_p1, g_p2])

    # H(P) = (I - beta G(P))^{-1}
    i3 = np.eye(3)
    h_p = np.linalg.inv(i3 - beta * g_p)

    # [Vhat(p0), Vhat(p1), Vhat(p2)]' = H(P) * (F(P)*m + G(P)*B')
    vhat_p = h_p @ (f_p * m + g_p @ b)

    t1_omega = omega @ p
    l_t1_omega, z_l_t1_omega = _compute_l_and_state(
        t1_omega, omega, p, b, beta, m, l_max, gamma, r2_omega
    )
    f_t1_omega, g_t1_omega = _f_and_g_from_l(l_t1_omega, z_l_t1_omega, beta)

    # Vhat(T1(omega)) = f(T1(omega))*m + g(T1(omega))*B' + beta*g(T1(omega))*Vhat(p)
    vhat_t1_omega = float(f_t1_omega * m + g_t1_omega @ b + beta * (g_t1_omega @ vhat_p))

    # Equation (37) residual:
    # omega*B' + beta*omega*Vhat(p) - m - beta*Vhat(T1(omega)) = 0.
    residual = float(omega @ b + beta * (omega @ vhat_p) - m - beta * vhat_t1_omega)

    details: Dict[str, object] = {
        "L(p0,omega)": l_p0,
        "L(p1,omega)": l_p1,
        "L(p2,omega)": l_p2,
        "L(T1(omega),omega)": l_t1_omega,
        "r2(omega)": r2_omega,
        "residual": residual,
    }
    return residual, details


def compute_single_arm_t2_whittle_index(
    omega: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    beta: float,
    config: SearchConfig = SearchConfig(),
) -> WhittleResult:
    omega = np.asarray(omega, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3, 3)
    b = np.asarray(b, dtype=float).reshape(3)
    _validate_inputs(omega, p, b, beta)

    a = 0.0
    c = float(b[2])  # search interval [0, B2]
    history: List[Dict[str, float]] = []

    last_grid = None
    last_residuals = None

    def refine_root(lo: float, hi: float) -> Tuple[float, float]:
        f_lo, _ = compute_residual(omega, p, b, beta, lo, config.l_max)
        f_hi, _ = compute_residual(omega, p, b, beta, hi, config.l_max)
        if abs(f_lo) <= config.eps:
            return float(lo), float(f_lo)
        if abs(f_hi) <= config.eps:
            return float(hi), float(f_hi)

        best_m = lo if abs(f_lo) <= abs(f_hi) else hi
        best_residual = f_lo if abs(f_lo) <= abs(f_hi) else f_hi
        a0, c0 = lo, hi

        for _ in range(config.bisect_steps):
            mid = 0.5 * (a0 + c0)
            f_mid, _ = compute_residual(omega, p, b, beta, mid, config.l_max)
            if abs(f_mid) < abs(best_residual):
                best_m = mid
                best_residual = f_mid
            if abs(f_mid) <= config.eps:
                return float(mid), float(f_mid)
            if f_lo * f_mid <= 0.0:
                c0 = mid
                f_hi = f_mid
            else:
                a0 = mid
                f_lo = f_mid

        return float(best_m), float(best_residual)

    for round_id in range(config.r_max + 1):
        j = config.j0 if round_id == 0 else config.j1
        grid = np.linspace(a, c, j + 1)
        residuals = np.zeros_like(grid)

        for idx, m in enumerate(grid):
            residual, _ = compute_residual(omega, p, b, beta, float(m), config.l_max)
            residuals[idx] = residual

        last_grid = grid
        last_residuals = residuals

        abs_residuals = np.abs(residuals)
        min_idx = int(np.argmin(abs_residuals))
        history.append(
            {
                "round": float(round_id),
                "a": float(a),
                "b": float(c),
                "best_m_on_grid": float(grid[min_idx]),
                "best_abs_residual_on_grid": float(abs_residuals[min_idx]),
            }
        )

        if abs_residuals[min_idx] <= config.eps:
            best_m = float(grid[min_idx])
            return WhittleResult(
                success=True,
                hat_w=best_m,
                residual=float(residuals[min_idx]),
                message="Converged by residual tolerance on grid.",
                interval=(float(a), float(c)),
                rounds=round_id + 1,
                history=history,
            )

        sign_change_intervals: List[Tuple[int, int]] = []
        for i in range(j):
            if residuals[i] * residuals[i + 1] < 0.0:
                sign_change_intervals.append((i, i + 1))

        if len(sign_change_intervals) == 1:
            i0, i1 = sign_change_intervals[0]
            root_m, root_residual = refine_root(float(grid[i0]), float(grid[i1]))
            if abs(root_residual) <= config.eps:
                return WhittleResult(
                    success=True,
                    hat_w=root_m,
                    residual=root_residual,
                    message="Converged by bisection after locating a sign-change interval.",
                    interval=(float(grid[i0]), float(grid[i1])),
                    rounds=round_id + 1,
                    history=history,
                )
            a, c = float(grid[i0]), float(grid[i1])
            continue

        if len(sign_change_intervals) > 1:
            return WhittleResult(
                success=False,
                hat_w=None,
                residual=None,
                message="More than one sign-change interval: relaxed indexability not numerically verified.",
                interval=(float(a), float(c)),
                rounds=round_id + 1,
                history=history,
            )

        # No sign change: shrink around the best grid point.
        if min_idx == 0:
            a, c = float(grid[0]), float(grid[1])
        elif min_idx == j:
            a, c = float(grid[j - 1]), float(grid[j])
        else:
            a, c = float(grid[min_idx - 1]), float(grid[min_idx + 1])

    final_idx = int(np.argmin(np.abs(last_residuals)))
    final_m = float(last_grid[final_idx])
    final_residual = float(last_residuals[final_idx])
    success = abs(final_residual) <= config.eps
    message = (
        "Reached max refinement rounds with acceptable residual."
        if success
        else "Reached max refinement rounds: relaxed indexability not numerically verified."
    )
    return WhittleResult(
        success=success,
        hat_w=(final_m if success else None),
        residual=final_residual,
        message=message,
        interval=(float(a), float(c)),
        rounds=config.r_max + 1,
        history=history,
    )


def compute_single_arm_finite_t2_whittle_index(
    omega: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    beta: float,
    eps: float = 1e-10,
    bisect_steps: int = 80,
) -> WhittleResult:
    """Compute the finite two-slot active/passive subsidy index.

    This solves

        omega B + beta omega max(PB, m) = m + beta max(omega P B, m)

    directly.  It is useful for finite-horizon plots where the benchmark is
    also a finite-horizon optimal value.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3, 3)
    b = np.asarray(b, dtype=float).reshape(3)
    _validate_inputs(omega, p, b, beta)

    lo = 0.0
    hi = float(b[2])

    def residual_at(m: float) -> float:
        gamma = _gamma_vector(p, b, m)
        return _r2_value(omega, p, b, beta, m, gamma)

    breakpoints = np.array(
        [lo, hi, float(omega @ b), float((omega @ p) @ b), *list(p @ b)],
        dtype=float,
    )
    breakpoints = np.clip(breakpoints, lo, hi)
    grid = np.unique(np.concatenate([np.linspace(lo, hi, 129), breakpoints]))
    grid.sort()
    residuals = np.array([residual_at(float(m)) for m in grid], dtype=float)
    abs_residuals = np.abs(residuals)
    min_idx = int(np.argmin(abs_residuals))

    history = [
        {
            "round": 0.0,
            "a": lo,
            "b": hi,
            "best_m_on_grid": float(grid[min_idx]),
            "best_abs_residual_on_grid": float(abs_residuals[min_idx]),
        }
    ]

    zero_grid = np.flatnonzero(abs_residuals <= eps)
    if zero_grid.size > 0:
        root_idx = int(zero_grid[-1])
        return WhittleResult(
            success=True,
            hat_w=float(grid[root_idx]),
            residual=float(residuals[root_idx]),
            message="Converged by finite two-slot residual tolerance on grid.",
            interval=(lo, hi),
            rounds=1,
            history=history,
        )

    candidates: List[Tuple[float, float]] = []
    for i in range(len(grid) - 1):
        f_lo = float(residuals[i])
        f_hi = float(residuals[i + 1])
        if f_lo * f_hi > 0.0:
            continue

        a0 = float(grid[i])
        c0 = float(grid[i + 1])
        best_m = a0 if abs(f_lo) <= abs(f_hi) else c0
        best_residual = f_lo if abs(f_lo) <= abs(f_hi) else f_hi

        for _ in range(bisect_steps):
            mid = 0.5 * (a0 + c0)
            f_mid = residual_at(mid)
            if abs(f_mid) < abs(best_residual):
                best_m = mid
                best_residual = f_mid
            if abs(f_mid) <= eps:
                best_m = mid
                best_residual = f_mid
                break
            if f_lo * f_mid <= 0.0:
                c0 = mid
                f_hi = f_mid
            else:
                a0 = mid
                f_lo = f_mid

        candidates.append((float(best_m), float(best_residual)))

    if candidates:
        acceptable = [item for item in candidates if abs(item[1]) <= eps]
        root_m, root_residual = (
            max(acceptable, key=lambda item: item[0])
            if acceptable
            else min(candidates, key=lambda item: abs(item[1]))
        )
        return WhittleResult(
            success=abs(root_residual) <= eps,
            hat_w=(root_m if abs(root_residual) <= eps else None),
            residual=root_residual,
            message=(
                "Converged by finite two-slot bisection."
                if abs(root_residual) <= eps
                else "Finite two-slot residual sign change did not refine to tolerance."
            ),
            interval=(lo, hi),
            rounds=1,
            history=history,
        )

    return WhittleResult(
        success=False,
        hat_w=None,
        residual=float(residuals[min_idx]),
        message="No finite two-slot root found on [0, B2].",
        interval=(lo, hi),
        rounds=1,
        history=history,
    )


def _demo() -> None:
    # Example from a 3-state setting.
    p = np.array(
        [
            [0.514, 0.321, 0.165],
            [0.123, 0.159, 0.718],
            [0.420, 0.195, 0.385],
        ],
        dtype=float,
    )
    b = np.array([0.0, 2.0, 3.0], dtype=float)
    omega = np.array([0.279, 0.618, 0.103], dtype=float)
    beta = 0.9999

    config = SearchConfig(l_max=350, j0=240, j1=120, r_max=8, eps=1e-7)
    result = compute_single_arm_t2_whittle_index(omega, p, b, beta, config)
    finite_result = compute_single_arm_finite_t2_whittle_index(omega, p, b, beta)

    print("success:", result.success)
    print("hat W(omega):", result.hat_w)
    print("residual:", result.residual)
    print("message:", result.message)
    print("finite t=2 W(omega):", finite_result.hat_w)

    if result.success and result.hat_w is not None:
        residual_check, detail = compute_residual(omega, p, b, beta, result.hat_w, config.l_max)
        print("post-check residual:", residual_check)
        print("L values:", detail["L(p0,omega)"], detail["L(p1,omega)"], detail["L(p2,omega)"], detail["L(T1(omega),omega)"])


if __name__ == "__main__":
    _demo()
