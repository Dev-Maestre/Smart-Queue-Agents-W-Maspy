import math
import random
from typing import Any

from config import (
    MIN_GREEN,
    QUEUE_CAP,
    SERVICE_RATE,
    SWITCH_PENALTY,
    PRIORITY_BONUS,
    PRIORITY_PENALTY,
    REWARD_THROUGHPUT_W,
    REWARD_QUEUE_LINEAR_W,
    REWARD_QUEUE_QUADRATIC_W,
    REWARD_MAX_QUEUE_W,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_K,
    ALPHA_START,
    ALPHA_END,
    ALPHA_DECAY_K,
)

DIRECTIONS = ("N", "S", "E", "W")
ACTIONS = ("hold", "N", "S", "E", "W", "NS", "EW")


def phase_allows(phase: str, direction: str) -> bool:
    return (
        phase == direction
        or (phase == "NS" and direction in {"N", "S"})
        or (phase == "EW" and direction in {"E", "W"})
    )


def served_counts(phase: str, qn: int, qs: int, qe: int, qw: int, service_rate: int = SERVICE_RATE):
    served_n = min(service_rate, qn) if phase in {"NS", "N"} else 0
    served_s = min(service_rate, qs) if phase in {"NS", "S"} else 0
    served_e = min(service_rate, qe) if phase in {"EW", "E"} else 0
    served_w = min(service_rate, qw) if phase in {"EW", "W"} else 0
    return served_n, served_s, served_e, served_w


def poisson(rng: random.Random, lam: float) -> int:
    # Knuth algorithm
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


def elapsed_green_bucket(time_step: int, last_switch_time: int, min_green: int = MIN_GREEN) -> int:
    elapsed = max(0, int(time_step) - int(last_switch_time))
    if elapsed < min_green:
        return 0
    if elapsed < 2 * min_green:
        return 1
    return 2


def pressure_bucket(qn: int, qs: int, qe: int, qw: int, queue_cap: int = QUEUE_CAP) -> int:
    pressure = (qn + qs) - (qe + qw)
    threshold = max(1, queue_cap // 2)
    if pressure <= -2 * threshold:
        return -2
    if pressure <= -threshold:
        return -1
    if pressure >= 2 * threshold:
        return 2
    if pressure >= threshold:
        return 1
    return 0


def add_derived_state_features(state: dict[str, Any], min_green: int = MIN_GREEN, queue_cap: int = QUEUE_CAP) -> dict[str, Any]:
    qn = int(state["queue_n"])
    qs = int(state["queue_s"])
    qe = int(state["queue_e"])
    qw = int(state["queue_w"])
    time_step = int(state["time"])
    last_switch = int(state.get("last_switch_time", 0))
    state["elapsed_green_bucket"] = elapsed_green_bucket(time_step, last_switch, min_green)
    state["pressure_bucket"] = pressure_bucket(qn, qs, qe, qw, queue_cap)
    return state


def compute_reward(
    prev_q: tuple[int, int, int, int],
    next_q: tuple[int, int, int, int],
    throughput: int,
    next_phase: str,
    switched: bool,
    switch_penalty: float = SWITCH_PENALTY,
) -> float:
    prev_n, prev_s, prev_e, prev_w = prev_q
    next_n, next_s, next_e, next_w = next_q
    queue_sum = next_n + next_s + next_e + next_w
    max_queue = max(next_n, next_s, next_e, next_w)

    reward = (
        REWARD_THROUGHPUT_W * throughput
        - REWARD_QUEUE_LINEAR_W * queue_sum
        - REWARD_QUEUE_QUADRATIC_W * (queue_sum ** 2)
        - REWARD_MAX_QUEUE_W * max_queue
    )
    if switched:
        reward -= switch_penalty

    max_dir = max(
        {"N": prev_n, "S": prev_s, "E": prev_e, "W": prev_w},
        key=lambda k: {"N": prev_n, "S": prev_s, "E": prev_e, "W": prev_w}[k],
    )
    reward += PRIORITY_BONUS if phase_allows(next_phase, max_dir) else -PRIORITY_PENALTY
    return reward


def epsilon_for_episode(ep_idx: int, total_episodes: int) -> float:
    frac = min(1.0, max(0.0, ep_idx / max(1, total_episodes - 1)))
    eps = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-EPSILON_DECAY_K * frac)
    return float(max(EPSILON_END, min(EPSILON_START, eps)))


def alpha_for_episode(ep_idx: int, total_episodes: int) -> float:
    frac = min(1.0, max(0.0, ep_idx / max(1, total_episodes - 1)))
    alpha = ALPHA_END + (ALPHA_START - ALPHA_END) * math.exp(-ALPHA_DECAY_K * frac)
    return float(max(ALPHA_END, min(ALPHA_START, alpha)))


def transition_step(
    state: dict[str, Any],
    action: str,
    rng: random.Random,
    lam_map: dict[str, float],
    *,
    min_green: int = MIN_GREEN,
    service_rate: int = SERVICE_RATE,
    queue_cap: int = QUEUE_CAP,
    yellow_duration: int = 2,
    switch_penalty: float = SWITCH_PENALTY,
):
    phase = str(state["phase"])
    qn = int(state["queue_n"])
    qs = int(state["queue_s"])
    qe = int(state["queue_e"])
    qw = int(state["queue_w"])
    t = int(state["time"])
    yellow = int(state["yellow"])
    last_switch_time = int(state.get("last_switch_time", 0))

    desired_action = action
    if desired_action != "hold" and t - last_switch_time < min_green:
        desired_action = "hold"
    if yellow > 0:
        desired_action = "hold"

    next_phase = phase if desired_action == "hold" else desired_action
    switched = next_phase != phase
    if switched and yellow == 0:
        next_yellow = yellow_duration
    else:
        next_yellow = max(0, yellow - 1)

    if next_yellow > 0:
        served_n = served_s = served_e = served_w = 0
    else:
        served_n, served_s, served_e, served_w = served_counts(next_phase, qn, qs, qe, qw, service_rate)

    arr_n = poisson(rng, lam_map["N"])
    arr_s = poisson(rng, lam_map["S"])
    arr_e = poisson(rng, lam_map["E"])
    arr_w = poisson(rng, lam_map["W"])

    next_qn = min(max(0, qn - served_n) + arr_n, queue_cap)
    next_qs = min(max(0, qs - served_s) + arr_s, queue_cap)
    next_qe = min(max(0, qe - served_e) + arr_e, queue_cap)
    next_qw = min(max(0, qw - served_w) + arr_w, queue_cap)
    next_t = t + 1
    throughput = served_n + served_s + served_e + served_w

    reward = compute_reward(
        (qn, qs, qe, qw),
        (next_qn, next_qs, next_qe, next_qw),
        throughput,
        next_phase,
        switched,
        switch_penalty=switch_penalty,
    )

    next_state = {
        "phase": next_phase,
        "queue_n": next_qn,
        "queue_s": next_qs,
        "queue_e": next_qe,
        "queue_w": next_qw,
        "time": next_t,
        "yellow": next_yellow,
        "last_switch_time": next_t if switched else last_switch_time,
    }
    add_derived_state_features(next_state, min_green=min_green, queue_cap=queue_cap)

    served = {"N": served_n, "S": served_s, "E": served_e, "W": served_w}
    return next_state, reward, throughput, served
