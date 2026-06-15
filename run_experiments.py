# -*- coding: latin-1 -*-

import statistics
from math import sqrt
import csv
from pathlib import Path
import random
from collections import defaultdict

import numpy as np

try:
    from tensorboardX import SummaryWriter  # type: ignore[import-not-found]
except Exception:
    SummaryWriter = None

from config import (
    SEEDS,
    TRAIN_EPISODES_LONG,
    HORIZON,
    RUN_LIGHT,
    ENABLE_TENSORBOARD,
    TB_LOG_DIR,
    REPLAY_EPISODES,
    ALPHA,
    GAMMA,
    MIN_GREEN,
    QUEUE_CAP,
    SERVICE_RATE,
    CONV_WINDOW,
    CONV_PATIENCE,
    CONV_STD_TOL,
    CONV_SLOPE_TOL,
    ADAPT_SHOCK_STEP,
    ADAPT_SHOCK_FACTOR,
    ADAPT_SHOCK_DIRECTION,
    ADAPT_EVAL_STEPS,
    ADAPT_RECOVERY_WINDOW,
    ADAPT_RECOVERY_RATIO,
    ABLATION_GAMMAS,
    ABLATION_EPISODES,
    ABLATION_MIN_THROUGHPUT_DROP,
)
from run_utils import get_run_dir, write_run_readme
from ex_smart_intersection import Intersection
from maspy.learning import EnvModel, qlearning
from maspy.learning.core import HashableWrapper
from maspy.admin import Admin

from sarsa_baseline import train_sarsa, TrafficEnv, epsilon_greedy
from rl_common import (
    ACTIONS,
    DIRECTIONS,
    add_derived_state_features,
    alpha_for_episode,
    epsilon_for_episode,
    transition_step,
)


def _ci95(values):
    if not values:
        return (0.0, 0.0, 0.0)
    mean = statistics.mean(values)
    if len(values) < 2:
        # Com uma unica amostra, IC degenerado no proprio valor medio.
        return (mean, mean, mean)
    sd = statistics.pstdev(values)
    half = 1.96 * sd / sqrt(len(values))
    return (mean, mean - half, mean + half)


def _select_action(model: EnvModel, state_tuple: tuple):
    st = HashableWrapper(state_tuple)
    q_values = model.q_table[st]
    action_idx = int(np.argmax(q_values))
    return model.actions_list[action_idx].original


def _state_to_tuple(state: dict):
    elapsed_green_bucket = int(state.get("elapsed_green_bucket", 0))
    pressure_bucket = int(state.get("pressure_bucket", 0))
    return (
        state["phase"],
        int(state["queue_n"]),
        int(state["queue_s"]),
        int(state["queue_e"]),
        int(state["queue_w"]),
        int(state["time"]),
        int(state["yellow"]),
        elapsed_green_bucket,
        pressure_bucket,
    )


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _greedy_from_q_table(q_table: dict, state_tuple: tuple):
    values = {a: q_table.get((state_tuple, a), 0.0) for a in ACTIONS}
    return max(values, key=values.get)


def _series_mean_std(series_by_seed: list[list[float]]):
    if not series_by_seed:
        return [], []
    length = min(len(s) for s in series_by_seed if s)
    if length == 0:
        return [], []
    means = []
    stds = []
    for i in range(length):
        vals = [s[i] for s in series_by_seed]
        means.append(float(statistics.mean(vals)))
        stds.append(float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0)
    return means, stds


def _detect_convergence_episode(rewards: list[float]):
    if len(rewards) < (2 * CONV_WINDOW + CONV_PATIENCE):
        return len(rewards)
    stable = 0
    for ep in range(CONV_WINDOW, len(rewards)):
        window_vals = rewards[ep - CONV_WINDOW + 1 : ep + 1]
        curr_mean = statistics.mean(window_vals)
        prev_vals = rewards[ep - CONV_WINDOW : ep]
        prev_mean = statistics.mean(prev_vals)
        curr_std = statistics.pstdev(window_vals) if len(window_vals) > 1 else 0.0

        if abs(curr_mean - prev_mean) <= CONV_SLOPE_TOL and curr_std <= CONV_STD_TOL:
            stable += 1
            if stable >= CONV_PATIENCE:
                return ep + 1
        else:
            stable = 0
    return len(rewards)


def _transition_with_directional_arrivals(state: dict, action: str, rng: random.Random, lam_map: dict[str, float]):
    new_state, reward, throughput, _ = transition_step(
        state,
        action,
        rng,
        lam_map,
        min_green=MIN_GREEN,
        service_rate=SERVICE_RATE,
        queue_cap=QUEUE_CAP,
        yellow_duration=2,
    )
    return new_state, reward, throughput


def _train_maspy_q_with_rewards(seed: int, train_episodes: int | None = None, gamma_override: float | None = None):
    rng = random.Random(seed)
    q_table = {}
    episode_rewards = []
    gamma = GAMMA if gamma_override is None else gamma_override
    episodes = TRAIN_EPISODES_LONG if train_episodes is None else train_episodes
    for ep in range(1, episodes + 1):
        state = {
            "phase": "NS",
            "queue_n": 0,
            "queue_s": 0,
            "queue_e": 0,
            "queue_w": 0,
            "time": 0,
            "yellow": 0,
            "last_switch_time": 0,
        }
        add_derived_state_features(state, min_green=MIN_GREEN, queue_cap=QUEUE_CAP)
        done = False
        ep_reward = 0.0
        eps = epsilon_for_episode(ep - 1, episodes)
        alpha = alpha_for_episode(ep - 1, episodes)
        while not done:
            st = _state_to_tuple(state)
            if rng.random() < eps:
                action = rng.choice(ACTIONS)
            else:
                action = _greedy_from_q_table(q_table, st)

            lam_map = {d: 1.0 for d in DIRECTIONS}
            next_state, reward, _ = _transition_with_directional_arrivals(state, action, rng, lam_map)
            next_st = _state_to_tuple(next_state)
            old_q = q_table.get((st, action), 0.0)
            next_max = max(q_table.get((next_st, a), 0.0) for a in ACTIONS)
            q_table[(st, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

            ep_reward += reward
            state = next_state
            done = int(state["time"]) >= HORIZON
        episode_rewards.append(ep_reward)
    return q_table, episode_rewards


def _train_baseline_with_rewards(seed: int, train_episodes: int | None = None, gamma_override: float | None = None):
    rng = random.Random(seed)
    env = TrafficEnv(rng)
    q_table = {}
    episode_rewards = []
    gamma = GAMMA if gamma_override is None else gamma_override
    episodes = TRAIN_EPISODES_LONG if train_episodes is None else train_episodes
    for ep in range(1, episodes + 1):
        state = env.reset()
        epsilon = epsilon_for_episode(ep - 1, episodes)
        alpha = alpha_for_episode(ep - 1, episodes)
        action = epsilon_greedy(q_table, state, epsilon, rng)
        done = False
        ep_reward = 0.0
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(q_table, next_state, epsilon, rng)
            old = q_table.get((state, action), 0.0)
            next_q = q_table.get((next_state, next_action), 0.0)
            q_table[(state, action)] = old + alpha * (reward + gamma * next_q - old)
            state, action = next_state, next_action
            ep_reward += reward
        episode_rewards.append(ep_reward)
    return q_table, episode_rewards


def _build_training_metrics(run_dir: Path):
    per_seed_conv_rows = []
    maspy_rewards_all = []
    baseline_rewards_all = []
    maspy_q_by_seed = {}
    baseline_q_by_seed = {}

    for seed in SEEDS:
        mq, mr = _train_maspy_q_with_rewards(seed)
        bq, br = _train_baseline_with_rewards(seed)

        maspy_q_by_seed[seed] = mq
        baseline_q_by_seed[seed] = bq
        maspy_rewards_all.append(mr)
        baseline_rewards_all.append(br)

        m_conv = _detect_convergence_episode(mr)
        b_conv = _detect_convergence_episode(br)
        per_seed_conv_rows.append(
            {
                "seed": seed,
                "maspy_convergence_episode": m_conv,
                "baseline_convergence_episode": b_conv,
            }
        )

    maspy_mean, maspy_std = _series_mean_std(maspy_rewards_all)
    baseline_mean, baseline_std = _series_mean_std(baseline_rewards_all)

    rewards_rows = []
    cum_m = 0.0
    cum_b = 0.0
    for ep in range(len(maspy_mean)):
        cum_m += maspy_mean[ep]
        cum_b += baseline_mean[ep]
        rewards_rows.append(
            {
                "episode": ep + 1,
                "maspy_reward_mean": round(maspy_mean[ep], 6),
                "maspy_reward_std": round(maspy_std[ep], 6),
                "baseline_reward_mean": round(baseline_mean[ep], 6),
                "baseline_reward_std": round(baseline_std[ep], 6),
                "maspy_cumulative_reward": round(cum_m, 6),
                "baseline_cumulative_reward": round(cum_b, 6),
            }
        )

    _write_csv(run_dir / "training_rewards_comparison.csv", rewards_rows)
    _write_csv(run_dir / "convergence_per_seed.csv", per_seed_conv_rows)

    m_vals = [int(r["maspy_convergence_episode"]) for r in per_seed_conv_rows]
    b_vals = [int(r["baseline_convergence_episode"]) for r in per_seed_conv_rows]
    m_ci = _ci95(m_vals)
    b_ci = _ci95(b_vals)
    conv_rows = [
        {
            "algorithm": "MASPY",
            "mean_convergence_episode": round(m_ci[0], 6),
            "ci95_low": round(m_ci[1], 6),
            "ci95_high": round(m_ci[2], 6),
        },
        {
            "algorithm": "Baseline",
            "mean_convergence_episode": round(b_ci[0], 6),
            "ci95_low": round(b_ci[1], 6),
            "ci95_high": round(b_ci[2], 6),
        },
    ]
    _write_csv(run_dir / "convergence_summary.csv", conv_rows)
    return maspy_q_by_seed, baseline_q_by_seed, per_seed_conv_rows


def _evaluate_q_table_greedy(seed: int, q_table: dict):
    rng = random.Random(seed + 404)
    state = {
        "phase": "NS",
        "queue_n": 0,
        "queue_s": 0,
        "queue_e": 0,
        "queue_w": 0,
        "time": 0,
        "yellow": 0,
        "last_switch_time": 0,
    }
    add_derived_state_features(state, min_green=MIN_GREEN, queue_cap=QUEUE_CAP)

    total_reward = 0.0
    total_throughput = 0.0
    steps = 0
    while int(state["time"]) < HORIZON:
        action = _greedy_from_q_table(q_table, _state_to_tuple(state))
        lam_map = {d: 1.0 for d in DIRECTIONS}
        next_state, reward, throughput = _transition_with_directional_arrivals(state, action, rng, lam_map)
        total_reward += reward
        total_throughput += throughput
        steps += 1
        state = next_state

    throughput_per_step = total_throughput / max(1, steps)
    return total_reward, throughput_per_step


def _run_gamma_ablation(run_dir: Path):
    rows = []
    gamma_values = [float(g) for g in ABLATION_GAMMAS]

    for gamma in gamma_values:
        maspy_rewards = []
        maspy_throughputs = []
        baseline_rewards = []
        baseline_throughputs = []

        for seed in SEEDS:
            mq, _ = _train_maspy_q_with_rewards(seed, train_episodes=ABLATION_EPISODES, gamma_override=gamma)
            bq, _ = _train_baseline_with_rewards(seed, train_episodes=ABLATION_EPISODES, gamma_override=gamma)

            m_reward, m_thr = _evaluate_q_table_greedy(seed, mq)
            b_reward, b_thr = _evaluate_q_table_greedy(seed, bq)
            maspy_rewards.append(m_reward)
            maspy_throughputs.append(m_thr)
            baseline_rewards.append(b_reward)
            baseline_throughputs.append(b_thr)

        rows.append(
            {
                "gamma": gamma,
                "maspy_reward_mean": round(statistics.mean(maspy_rewards), 6),
                "maspy_throughput_mean": round(statistics.mean(maspy_throughputs), 6),
                "baseline_reward_mean": round(statistics.mean(baseline_rewards), 6),
                "baseline_throughput_mean": round(statistics.mean(baseline_throughputs), 6),
            }
        )

    _write_csv(run_dir / "ablation_gamma_summary.csv", rows)

    ref = next((r for r in rows if abs(float(r["gamma"]) - 0.9) < 1e-9), rows[0])
    maspy_thr_ref = float(ref["maspy_throughput_mean"])
    baseline_thr_ref = float(ref["baseline_throughput_mean"])
    maspy_candidates = [
        r
        for r in rows
        if float(r["maspy_throughput_mean"]) >= (maspy_thr_ref - ABLATION_MIN_THROUGHPUT_DROP)
    ]
    baseline_candidates = [
        r
        for r in rows
        if float(r["baseline_throughput_mean"]) >= (baseline_thr_ref - ABLATION_MIN_THROUGHPUT_DROP)
    ]

    best_maspy = max(maspy_candidates or rows, key=lambda r: float(r["maspy_reward_mean"]))
    best_baseline = max(baseline_candidates or rows, key=lambda r: float(r["baseline_reward_mean"]))

    best_rows = [
        {
            "algorithm": "MASPY",
            "best_gamma": best_maspy["gamma"],
            "reward_mean": best_maspy["maspy_reward_mean"],
            "throughput_mean": best_maspy["maspy_throughput_mean"],
        },
        {
            "algorithm": "Baseline",
            "best_gamma": best_baseline["gamma"],
            "reward_mean": best_baseline["baseline_reward_mean"],
            "throughput_mean": best_baseline["baseline_throughput_mean"],
        },
    ]
    _write_csv(run_dir / "ablation_gamma_best.csv", best_rows)
    return best_rows


def _simulate_adaptation_for_algo(seed: int, algo: str, q_table: dict):
    rng = random.Random(seed + 2026)
    state = {
        "phase": "NS",
        "queue_n": 0,
        "queue_s": 0,
        "queue_e": 0,
        "queue_w": 0,
        "time": 0,
        "yellow": 0,
        "last_switch_time": 0,
    }
    add_derived_state_features(state, min_green=MIN_GREEN, queue_cap=QUEUE_CAP)
    rows = []
    pre_shock_throughputs = []
    recovery_step = -1
    for step in range(1, ADAPT_EVAL_STEPS + 1):
        state_tuple = _state_to_tuple(state)
        action = _greedy_from_q_table(q_table, state_tuple)
        lam_map = {d: 1.0 for d in DIRECTIONS}
        if step >= ADAPT_SHOCK_STEP:
            lam_map[ADAPT_SHOCK_DIRECTION] = 1.0 * ADAPT_SHOCK_FACTOR
        next_state, reward, throughput = _transition_with_directional_arrivals(state, action, rng, lam_map)

        queue_sum = (
            int(next_state["queue_n"])
            + int(next_state["queue_s"])
            + int(next_state["queue_e"])
            + int(next_state["queue_w"])
        )
        rows.append(
            {
                "seed": seed,
                "algorithm": algo,
                "step": step,
                "shock_active": 1 if step >= ADAPT_SHOCK_STEP else 0,
                "phase": next_state["phase"],
                "queue_sum": queue_sum,
                "throughput": throughput,
                "reward": round(reward, 6),
                "arrival_lambda_N": lam_map["N"],
                "arrival_lambda_S": lam_map["S"],
                "arrival_lambda_E": lam_map["E"],
                "arrival_lambda_W": lam_map["W"],
            }
        )

        if step < ADAPT_SHOCK_STEP:
            pre_shock_throughputs.append(throughput)
        elif recovery_step < 0 and len(rows) >= ADAPT_RECOVERY_WINDOW:
            baseline_thr = statistics.mean(pre_shock_throughputs[-ADAPT_RECOVERY_WINDOW:]) if pre_shock_throughputs else 0.0
            recent = [r["throughput"] for r in rows[-ADAPT_RECOVERY_WINDOW:]]
            recent_avg = statistics.mean(recent)
            if baseline_thr > 0 and recent_avg >= baseline_thr * ADAPT_RECOVERY_RATIO:
                recovery_step = step - ADAPT_SHOCK_STEP + 1

        state = next_state

    if recovery_step < 0:
        recovery_step = ADAPT_EVAL_STEPS - ADAPT_SHOCK_STEP + 1
    return rows, recovery_step


def _build_adaptation_metrics(run_dir: Path, maspy_q_by_seed: dict, baseline_q_by_seed: dict):
    rows = []
    recovery_rows = []
    for seed in SEEDS:
        m_rows, m_rec = _simulate_adaptation_for_algo(seed, "MASPY", maspy_q_by_seed[seed])
        b_rows, b_rec = _simulate_adaptation_for_algo(seed, "Baseline", baseline_q_by_seed[seed])
        rows.extend(m_rows)
        rows.extend(b_rows)
        recovery_rows.append({"seed": seed, "algorithm": "MASPY", "recovery_steps": m_rec})
        recovery_rows.append({"seed": seed, "algorithm": "Baseline", "recovery_steps": b_rec})

    _write_csv(run_dir / "adaptation_shock_timeseries.csv", rows)
    _write_csv(run_dir / "adaptation_recovery_per_seed.csv", recovery_rows)

    grouped = defaultdict(list)
    for r in recovery_rows:
        grouped[r["algorithm"]].append(int(r["recovery_steps"]))
    summary_rows = []
    for algo in ("MASPY", "Baseline"):
        vals = grouped.get(algo, [])
        ci = _ci95(vals)
        summary_rows.append(
            {
                "algorithm": algo,
                "mean_recovery_steps": round(ci[0], 6),
                "ci95_low": round(ci[1], 6),
                "ci95_high": round(ci[2], 6),
            }
        )
    _write_csv(run_dir / "adaptation_recovery_summary.csv", summary_rows)


def _rollout_maspy_episode(env: Intersection, model: EnvModel, seed: int, ep: int, writer=None, collect_rows: bool = False, track_metrics: bool = True):
    state = {
        "phase": "NS",
        "queue_n": 0,
        "queue_s": 0,
        "queue_e": 0,
        "queue_w": 0,
        "time": 0,
        "yellow": 0,
        "last_switch_time": 0,
    }
    add_derived_state_features(state, min_green=MIN_GREEN, queue_cap=QUEUE_CAP)
    rows = []
    queue_sums = []
    throughput_total = 0
    steps = 0
    done = False
    while not done:
        state_tuple = _state_to_tuple(state)
        action = _select_action(model, state_tuple)
        new_state, _, done = env.control_transition(state, action)
        served = env._served_counts(
            new_state["phase"],
            int(state["queue_n"]),
            int(state["queue_s"]),
            int(state["queue_e"]),
            int(state["queue_w"]),
        )
        queue_sum = (
            int(new_state["queue_n"])
            + int(new_state["queue_s"])
            + int(new_state["queue_e"])
            + int(new_state["queue_w"])
        )
        if collect_rows:
            rows.append(
                {
                    "episode": ep,
                    "time": int(new_state["time"]),
                    "phase": new_state["phase"],
                    "queue_sum": queue_sum,
                    "q_n": int(new_state["queue_n"]),
                    "q_s": int(new_state["queue_s"]),
                    "q_e": int(new_state["queue_e"]),
                    "q_w": int(new_state["queue_w"]),
                    "yellow": int(new_state["yellow"]),
                    "served_n": served[0],
                    "served_s": served[1],
                    "served_e": served[2],
                    "served_w": served[3],
                }
            )
        if track_metrics:
            throughput_total += sum(served)
            queue_sums.append(queue_sum)
            if writer is not None:
                writer.add_scalar(f"maspy/seed_{seed}/queue_sum", queue_sum, steps + 1)
                writer.add_scalar(
                    f"maspy/seed_{seed}/throughput_per_step",
                    throughput_total / (steps + 1),
                    steps + 1,
                )
            steps += 1
        state = new_state
        if state["time"] >= HORIZON:
            break
    return rows, queue_sums, throughput_total, steps


def _rollout_maspy_q_episode(seed: int, q_table: dict, ep: int, writer=None, collect_rows: bool = False, track_metrics: bool = True):
    rng = random.Random(seed + 1000 + ep)
    state = {
        "phase": "NS",
        "queue_n": 0,
        "queue_s": 0,
        "queue_e": 0,
        "queue_w": 0,
        "time": 0,
        "yellow": 0,
        "last_switch_time": 0,
    }
    add_derived_state_features(state, min_green=MIN_GREEN, queue_cap=QUEUE_CAP)

    rows = []
    queue_sums = []
    throughput_total = 0
    steps = 0

    while int(state["time"]) < HORIZON:
        state_tuple = _state_to_tuple(state)
        action = _greedy_from_q_table(q_table, state_tuple)
        lam_map = {d: 1.0 for d in DIRECTIONS}
        new_state, _, _, served = transition_step(
            state,
            action,
            rng,
            lam_map,
            min_green=MIN_GREEN,
            service_rate=SERVICE_RATE,
            queue_cap=QUEUE_CAP,
            yellow_duration=2,
        )

        queue_sum = (
            int(new_state["queue_n"])
            + int(new_state["queue_s"])
            + int(new_state["queue_e"])
            + int(new_state["queue_w"])
        )

        if collect_rows:
            rows.append(
                {
                    "episode": ep,
                    "time": int(new_state["time"]),
                    "phase": new_state["phase"],
                    "queue_sum": queue_sum,
                    "q_n": int(new_state["queue_n"]),
                    "q_s": int(new_state["queue_s"]),
                    "q_e": int(new_state["queue_e"]),
                    "q_w": int(new_state["queue_w"]),
                    "yellow": int(new_state["yellow"]),
                    "served_n": int(served["N"]),
                    "served_s": int(served["S"]),
                    "served_e": int(served["E"]),
                    "served_w": int(served["W"]),
                }
            )

        if track_metrics:
            throughput_total += int(served["N"]) + int(served["S"]) + int(served["E"]) + int(served["W"])
            queue_sums.append(queue_sum)
            if writer is not None:
                writer.add_scalar(f"maspy/seed_{seed}/queue_sum", queue_sum, steps + 1)
                writer.add_scalar(
                    f"maspy/seed_{seed}/throughput_per_step",
                    throughput_total / (steps + 1),
                    steps + 1,
                )
            steps += 1

        state = new_state

    return rows, queue_sums, throughput_total, steps


def simulate_maspy(seed: int, writer=None, collect_rows: bool = False, record_episodes: int = 1):
    if RUN_LIGHT:
        q_table, _ = _train_maspy_q_with_rewards(seed, train_episodes=TRAIN_EPISODES_LONG)
        rows = []
        episode_means = []
        episode_throughputs = []
        steps = 0
        total_eps = record_episodes if collect_rows else 1
        for ep in range(total_eps):
            ep_rows, ep_queue_sums, ep_throughput, ep_steps = _rollout_maspy_q_episode(
                seed,
                q_table,
                ep,
                writer=writer,
                collect_rows=collect_rows,
                track_metrics=(ep == 0),
            )
            rows.extend(ep_rows)
            if ep_queue_sums:
                episode_means.append(statistics.mean(ep_queue_sums))
            if ep_steps > 0:
                episode_throughputs.append(ep_throughput / ep_steps)
            if ep == 0:
                steps = ep_steps

        return {
            "steps": steps,
            "avg_queue_sum": statistics.mean(episode_means) if episode_means else 0.0,
            "throughput_per_step": statistics.mean(episode_throughputs) if episode_throughputs else 0.0,
            "rows": rows,
        }

    random.seed(seed)
    np.random.seed(seed)
    Admin().reset_instance()
    env = Intersection(f"I1_{seed}")
    env.horizon = HORIZON
    env._states["time"] = list(range(HORIZON + 1))

    model = EnvModel(env)
    model.learn(
        qlearning,
        num_episodes=TRAIN_EPISODES_LONG,
        max_steps=HORIZON,
        epsilon=1.0,
        final_epsilon=0.1,
    )

    rows = []
    episode_means = []
    episode_throughputs = []
    steps = 0
    total_eps = record_episodes if collect_rows else 1
    for ep in range(total_eps):
        ep_rows, ep_queue_sums, ep_throughput, ep_steps = _rollout_maspy_episode(
            env,
            model,
            seed,
            ep,
            writer=writer,
            collect_rows=collect_rows,
            track_metrics=(ep == 0),
        )
        rows.extend(ep_rows)
        if ep_queue_sums:
            episode_means.append(statistics.mean(ep_queue_sums))
        if ep_steps > 0:
            episode_throughputs.append(ep_throughput / ep_steps)
        if ep == 0:
            steps = ep_steps
    return {
        "steps": steps,
        "avg_queue_sum": statistics.mean(episode_means) if episode_means else 0.0,
        "throughput_per_step": statistics.mean(episode_throughputs) if episode_throughputs else 0.0,
        "rows": rows,
    }


def simulate_maspy_pair(seed: int, writer=None, collect_rows: bool = False, record_episodes: int = 1):
    # Simula duas intersecoes com o mesmo seed para gerar I1/I2 na mesma execucao
    m1 = simulate_maspy(seed, writer, collect_rows=collect_rows, record_episodes=record_episodes)
    m2 = simulate_maspy(seed + 1, writer, collect_rows=collect_rows, record_episodes=record_episodes)
    return m1, m2


def _rollout_baseline_episode(env: TrafficEnv, q_table, seed: int, ep: int, writer=None, collect_rows: bool = False, track_metrics: bool = True):
    env.reset()
    rows = []
    queue_sums = []
    throughput_total = 0
    done = False
    while not done:
        state = env.state()
        action = epsilon_greedy(q_table, state, 0.0, env.rng)
        _, _, done, info = env.step(action)
        if collect_rows:
            rows.append(
                {
                    "episode": ep,
                    "time": env.t,
                    "phase": env.phase,
                    "queue_sum": info["queue_sum"],
                    "q_n": env.q["N"],
                    "q_s": env.q["S"],
                    "q_e": env.q["E"],
                    "q_w": env.q["W"],
                    "yellow": env.yellow,
                    "served_n": info["served"]["N"],
                    "served_s": info["served"]["S"],
                    "served_e": info["served"]["E"],
                    "served_w": info["served"]["W"],
                }
            )
        if track_metrics:
            queue_sums.append(info["queue_sum"])
            throughput_total += sum(info["served"].values())
            if writer is not None:
                step_idx = len(queue_sums)
                writer.add_scalar(f"baseline/seed_{seed}/queue_sum", info["queue_sum"], step_idx)
                writer.add_scalar(
                    f"baseline/seed_{seed}/throughput_per_step",
                    throughput_total / step_idx,
                    step_idx,
                )
    return rows, queue_sums, throughput_total


def simulate_baseline(seed: int, writer=None, collect_rows: bool = False, record_episodes: int = 1):
    q_table = train_sarsa(seed, TRAIN_EPISODES_LONG, tb_writer=writer, tb_tag=f"baseline/seed_{seed}")
    env = TrafficEnv(random.Random(seed))
    rows = []
    episode_means = []
    episode_throughputs = []

    total_eps = record_episodes if collect_rows else 1
    for ep in range(total_eps):
        ep_rows, ep_queue_sums, ep_throughput = _rollout_baseline_episode(
            env,
            q_table,
            seed,
            ep,
            writer=writer,
            collect_rows=collect_rows,
            track_metrics=(ep == 0),
        )
        rows.extend(ep_rows)
        if ep_queue_sums:
            episode_means.append(statistics.mean(ep_queue_sums))
        if ep_queue_sums:
            episode_throughputs.append(ep_throughput / len(ep_queue_sums))
    steps = len(ep_queue_sums) if "ep_queue_sums" in locals() else 0
    return {
        "steps": steps,
        "avg_queue_sum": statistics.mean(episode_means) if episode_means else 0.0,
        "throughput_per_step": statistics.mean(episode_throughputs) if episode_throughputs else 0.0,
        "rows": rows,
    }


def simulate_baseline_pair(seed: int, writer=None, collect_rows: bool = False, record_episodes: int = 1):
    b1 = simulate_baseline(seed, writer, collect_rows=collect_rows, record_episodes=record_episodes)
    b2 = simulate_baseline(seed + 1, writer, collect_rows=collect_rows, record_episodes=record_episodes)
    return b1, b2


def _write_timeseries_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    tb_writer = None
    run_dir = get_run_dir("experiments")
    write_run_readme(
        run_dir,
        {
            "type": "experiments",
            "seeds": SEEDS,
            "train_episodes_long": TRAIN_EPISODES_LONG,
            "horizon": HORIZON,
        },
    )
    tb_dir = run_dir / "tensorboard"
    if ENABLE_TENSORBOARD and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
    maspy_queue = []
    maspy_throughput = []
    baseline_queue = []
    baseline_throughput = []

    per_seed_rows = []
    for idx, seed in enumerate(SEEDS):
        collect = idx == 0
        m1, m2 = simulate_maspy_pair(seed, tb_writer, collect_rows=collect, record_episodes=REPLAY_EPISODES)
        b1, b2 = simulate_baseline_pair(seed, tb_writer, collect_rows=collect, record_episodes=REPLAY_EPISODES)
        maspy_queue.append(m1["avg_queue_sum"])
        maspy_throughput.append(m1["throughput_per_step"])
        baseline_queue.append(b1["avg_queue_sum"])
        baseline_throughput.append(b1["throughput_per_step"])
        per_seed_rows.append(
            {
                "seed": seed,
                "maspy_avg_queue": m1["avg_queue_sum"],
                "maspy_throughput": m1["throughput_per_step"],
                "baseline_avg_queue": b1["avg_queue_sum"],
                "baseline_throughput": b1["throughput_per_step"],
            }
        )
        if collect:
            _write_timeseries_csv(run_dir / "metrics_I1.csv", m1["rows"])
            _write_timeseries_csv(run_dir / "metrics_I2.csv", m2["rows"])
            _write_timeseries_csv(run_dir / "metrics_baseline_I1.csv", b1["rows"])
            _write_timeseries_csv(run_dir / "metrics_baseline_I2.csv", b2["rows"])

    maspy_q_by_seed, baseline_q_by_seed, conv_rows = _build_training_metrics(run_dir)
    _build_adaptation_metrics(run_dir, maspy_q_by_seed, baseline_q_by_seed)
    best_gamma_rows = _run_gamma_ablation(run_dir)

    m_q = _ci95(maspy_queue)
    m_t = _ci95(maspy_throughput)
    b_q = _ci95(baseline_queue)
    b_t = _ci95(baseline_throughput)

    report = run_dir / "relatorio_experimentos.md"
    lines = []
    lines.append("# Relatorio de Experimentos (10 seeds)")
    lines.append("")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Episodios de treino: {TRAIN_EPISODES_LONG}")
    lines.append(f"- Horizonte de avaliacao: {HORIZON}")
    lines.append("")
    lines.append("## MASPY (media +/- IC 95%)")
    lines.append(
        f"- Fila media total: {m_q[0]:.2f} (IC95% [{m_q[1]:.2f}, {m_q[2]:.2f}])"
    )
    lines.append(
        f"- Throughput por passo: {m_t[0]:.2f} (IC95% [{m_t[1]:.2f}, {m_t[2]:.2f}])"
    )
    lines.append("")
    lines.append("## Baseline SARSA (media +/- IC 95%)")
    lines.append(
        f"- Fila media total: {b_q[0]:.2f} (IC95% [{b_q[1]:.2f}, {b_q[2]:.2f}])"
    )
    lines.append(
        f"- Throughput por passo: {b_t[0]:.2f} (IC95% [{b_t[1]:.2f}, {b_t[2]:.2f}])"
    )
    lines.append("")
    lines.append("## Novas metricas de RL")
    lines.append("")
    if conv_rows:
        m_conv = statistics.mean([int(r["maspy_convergence_episode"]) for r in conv_rows])
        b_conv = statistics.mean([int(r["baseline_convergence_episode"]) for r in conv_rows])
        lines.append(f"- Convergencia media (episodios): MASPY={m_conv:.2f} | Baseline={b_conv:.2f}")
    lines.append("- Recompensa cumulativa por episodio em: training_rewards_comparison.csv")
    lines.append("- Adaptacao a choque de demanda em: adaptation_shock_timeseries.csv")
    lines.append("- Recuperacao pos-choque em: adaptation_recovery_summary.csv")
    lines.append("- Ablation de gamma em: ablation_gamma_summary.csv")
    for r in best_gamma_rows:
        lines.append(
            f"- Melhor gamma {r['algorithm']}: {r['best_gamma']} (reward={float(r['reward_mean']):.2f}, throughput={float(r['throughput_mean']):.2f})"
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    summary_csv = run_dir / "experimentos_summary.csv"
    with summary_csv.open("w", newline="") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "maspy_avg_queue",
                "maspy_throughput",
                "baseline_avg_queue",
                "baseline_throughput",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(per_seed_rows)
    print(f"Relatorio gerado: {report}")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()

