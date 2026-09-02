from functools import lru_cache
from itertools import product
from typing import Callable, Tuple

import numpy as np


IndexFunction = Callable[[np.ndarray], np.ndarray]


def _reward_matrix(reward_set: np.ndarray, arm_num: int) -> np.ndarray:
    reward = np.asarray(reward_set, dtype=float)
    if reward.ndim == 1:
        reward = np.array([reward for _ in range(arm_num)], dtype=float)
    return reward


def expected_index_policy_curve(
    index_function: IndexFunction,
    arm_num: int,
    state_num: int,
    initial_dist: np.ndarray,
    transition_matrics: np.ndarray,
    active_num: int,
    reward_set: np.ndarray,
    time: int,
    round_decimals: int = 12,
    return_expected_rewards: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Evaluate an index policy by exact expectation instead of Monte Carlo.

    The Markov chains are uncontrolled, so the next hidden state distribution
    can be integrated exactly.  This removes simulation noise when comparing a
    policy curve to the finite-horizon optimal benchmark.
    """
    reward = _reward_matrix(reward_set, arm_num)
    initial = np.asarray(initial_dist, dtype=float).reshape(arm_num, state_num)
    p = np.asarray(transition_matrics, dtype=float).reshape(arm_num, state_num, state_num)

    def pack(belief: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(belief.reshape(-1), round_decimals))

    def unpack(key: Tuple[float, ...]) -> np.ndarray:
        return np.array(key, dtype=float).reshape(arm_num, state_num)

    @lru_cache(maxsize=None)
    def recurse(t: int, belief_key: Tuple[float, ...]) -> Tuple[float, ...]:
        belief = unpack(belief_key)
        indices = np.asarray(index_function(belief.copy()), dtype=float).reshape(arm_num)
        active = tuple(indices.argsort()[::-1][:active_num])
        active_set = set(active)

        current_reward = float(sum(belief[i] @ reward[i] for i in active))
        if t == time - 1:
            return (current_reward,)

        base_next = belief.copy()
        for i in range(arm_num):
            if i not in active_set:
                base_next[i] = belief[i] @ p[i]

        future = np.zeros(time - t - 1, dtype=float)
        for observed_states in product(range(state_num), repeat=active_num):
            prob = 1.0
            next_belief = base_next.copy()
            for arm, state in zip(active, observed_states):
                prob *= belief[arm, state]
                next_belief[arm] = p[arm, state]
            if prob > 0.0:
                future += prob * np.array(recurse(t + 1, pack(next_belief)))

        return tuple(np.concatenate([[current_reward], future]))

    expected_rewards = np.array(recurse(0, pack(initial)), dtype=float)
    curve = expected_rewards.cumsum() / np.arange(1, time + 1)

    if return_expected_rewards:
        return curve, expected_rewards
    return curve
