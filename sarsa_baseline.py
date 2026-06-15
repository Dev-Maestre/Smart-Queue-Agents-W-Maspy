import csv
import statistics
from pathlib import Path
from random import Random

from config import (
    SEED,
    TRAIN_EPISODES,
    HORIZON,
    ALPHA,
    GAMMA,
    EPSILON,
    MIN_GREEN,
    QUEUE_CAP,
    SERVICE_RATE,
    ARR_MIN,
    ARR_MAX,
    SWITCH_PENALTY,
    QUEUE_PENALTY_W,
    PRIORITY_BONUS,
    PRIORITY_PENALTY,
    PAIRED_ARRIVALS,
    ARRIVAL_SEED,
    ENABLE_TENSORBOARD,
    TB_LOG_DIR,
)
from run_utils import get_run_dir, write_run_readme
from rl_common import (
    ACTIONS,
    DIRECTIONS,
    add_derived_state_features,
    alpha_for_episode,
    epsilon_for_episode,
    transition_step,
)


# Configuracoes principais (padronizadas em config.py)
EVAL_HORIZON = HORIZON


class TrafficEnv:
    def __init__(self, rng: Random):
        self.rng = rng
        self.queue_cap = QUEUE_CAP
        self.service_rate = SERVICE_RATE
        self.arr_min = ARR_MIN
        self.arr_max = ARR_MAX
        self.arr_lambda = 1.0
        self.switch_penalty = SWITCH_PENALTY
        self.queue_penalty_w = QUEUE_PENALTY_W
        self.priority_bonus = PRIORITY_BONUS
        self.priority_penalty = PRIORITY_PENALTY
        self.min_green = MIN_GREEN
        self.yellow_duration = 2
        self.reset()

    def reset(self):
        self.state_dict = {
            "phase": "NS",
            "queue_n": 0,
            "queue_s": 0,
            "queue_e": 0,
            "queue_w": 0,
            "time": 0,
            "yellow": 0,
            "last_switch_time": 0,
        }
        add_derived_state_features(self.state_dict, min_green=self.min_green, queue_cap=self.queue_cap)
        self._sync_fields_from_state_dict()
        return self.state()

    def _sync_fields_from_state_dict(self):
        self.phase = str(self.state_dict["phase"])
        self.q = {
            "N": int(self.state_dict["queue_n"]),
            "S": int(self.state_dict["queue_s"]),
            "E": int(self.state_dict["queue_e"]),
            "W": int(self.state_dict["queue_w"]),
        }
        self.t = int(self.state_dict["time"])
        self.yellow = int(self.state_dict["yellow"])
        self.last_switch_time = int(self.state_dict["last_switch_time"])

    def state(self):
        return (
            self.phase,
            self.q["N"],
            self.q["S"],
            self.q["E"],
            self.q["W"],
            self.t,
            self.yellow,
            int(self.state_dict.get("elapsed_green_bucket", 0)),
            int(self.state_dict.get("pressure_bucket", 0)),
        )

    def step(self, action):
        lam_map = {d: self.arr_lambda for d in DIRECTIONS}
        next_state, reward, throughput, served = transition_step(
            self.state_dict,
            action,
            self.rng,
            lam_map,
            min_green=self.min_green,
            service_rate=self.service_rate,
            queue_cap=self.queue_cap,
            yellow_duration=self.yellow_duration,
            switch_penalty=self.switch_penalty,
        )
        self.state_dict = next_state
        self._sync_fields_from_state_dict()
        queue_sum = sum(self.q.values())

        done = self.t >= EVAL_HORIZON
        info = {"served": served, "queue_sum": queue_sum}
        return self.state(), reward, done, info


def epsilon_greedy(q_table, state, epsilon, rng: Random):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    values = {a: q_table.get((state, a), 0.0) for a in ACTIONS}
    return max(values, key=values.get)


def train_sarsa(
    seed: int,
    train_episodes: int = TRAIN_EPISODES,
    tb_writer=None,
    tb_tag: str = "baseline",
    gamma_override: float | None = None,
):
    rng = Random(seed)
    env = TrafficEnv(rng)
    q_table = {}
    gamma = GAMMA if gamma_override is None else gamma_override
    for ep in range(1, train_episodes + 1):
        state = env.reset()
        epsilon = epsilon_for_episode(ep - 1, train_episodes)
        alpha = alpha_for_episode(ep - 1, train_episodes)
        action = epsilon_greedy(q_table, state, epsilon, rng)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(q_table, next_state, epsilon, rng)
            old = q_table.get((state, action), 0.0)
            next_q = q_table.get((next_state, next_action), 0.0)
            q_table[(state, action)] = old + alpha * (reward + gamma * next_q - old)
            state, action = next_state, next_action
            ep_reward += reward
            ep_steps += 1
        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/episode_reward", ep_reward, ep)
            tb_writer.add_scalar(f"{tb_tag}/episode_steps", ep_steps, ep)
    return q_table


def evaluate(q_table, output_csv, seed: int):
    rng = Random(seed)
    env = TrafficEnv(rng)
    env.reset()
    rows = []
    done = False
    while not done:
        state = env.state()
        action = epsilon_greedy(q_table, state, 0.0, rng)
        _, reward, done, info = env.step(action)
        rows.append(
            {
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
                "reward": round(reward, 3),
                "action": action,
            }
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "phase",
                "queue_sum",
                "q_n",
                "q_s",
                "q_e",
                "q_w",
                "served_n",
                "served_s",
                "served_e",
                "served_w",
                "yellow",
                "reward",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    queue_sum = [int(r["queue_sum"]) for r in rows]
    throughput = {
        "N": sum(int(r["served_n"]) for r in rows),
        "S": sum(int(r["served_s"]) for r in rows),
        "E": sum(int(r["served_e"]) for r in rows),
        "W": sum(int(r["served_w"]) for r in rows),
    }
    return {
        "steps": len(rows),
        "avg_queue_sum": statistics.mean(queue_sum),
        "max_queue_sum": max(queue_sum),
        "throughput_total": sum(throughput.values()),
        "throughput_per_step": sum(throughput.values()) / len(rows),
    }


def summarize_maspy(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    queue_sum = [int(r["queue_sum"]) for r in rows]
    throughput = {
        "N": sum(int(r["served_n"]) for r in rows),
        "S": sum(int(r["served_s"]) for r in rows),
        "E": sum(int(r["served_e"]) for r in rows),
        "W": sum(int(r["served_w"]) for r in rows),
    }
    return {
        "steps": len(rows),
        "avg_queue_sum": statistics.mean(queue_sum),
        "max_queue_sum": max(queue_sum),
        "throughput_total": sum(throughput.values()),
        "throughput_per_step": sum(throughput.values()) / len(rows),
    }


def write_comparison_report(base_dir: Path, baseline_csv_i1: Path, baseline_csv_i2: Path):
    report = base_dir / "relatorio_comparativo_baseline.md"
    base_summary_i1 = summarize(baseline_csv_i1)
    base_summary_i2 = summarize(baseline_csv_i2)
    maspy_i1 = summarize_maspy(base_dir / "metrics_I1.csv") if (base_dir / "metrics_I1.csv").exists() else {}
    maspy_i2 = summarize_maspy(base_dir / "metrics_I2.csv") if (base_dir / "metrics_I2.csv").exists() else {}

    lines = []
    lines.append("# Comparativo - MASPY vs SARSA (Python puro)")
    lines.append("")
    lines.append("## MASPY (Intersecoes)")
    if maspy_i1:
        lines.append(
            f"- I1: passos={maspy_i1['steps']} | fila_media={maspy_i1['avg_queue_sum']:.2f} | "
            f"max_fila={maspy_i1['max_queue_sum']} | throughput={maspy_i1['throughput_total']} | "
            f"por_passo={maspy_i1['throughput_per_step']:.2f}"
        )
    if maspy_i2:
        lines.append(
            f"- I2: passos={maspy_i2['steps']} | fila_media={maspy_i2['avg_queue_sum']:.2f} | "
            f"max_fila={maspy_i2['max_queue_sum']} | throughput={maspy_i2['throughput_total']} | "
            f"por_passo={maspy_i2['throughput_per_step']:.2f}"
        )
    if not maspy_i1 and not maspy_i2:
        lines.append("- Sem dados MASPY (execute o simulador MASPY para gerar metrics_I1.csv/I2.csv).")
    lines.append("")
    lines.append("## Baseline (Python puro - SARSA)")
    if base_summary_i1:
        lines.append(
            f"- I1: passos={base_summary_i1['steps']} | fila_media={base_summary_i1['avg_queue_sum']:.2f} | "
            f"max_fila={base_summary_i1['max_queue_sum']} | throughput={base_summary_i1['throughput_total']} | "
            f"por_passo={base_summary_i1['throughput_per_step']:.2f}"
        )
    if base_summary_i2:
        lines.append(
            f"- I2: passos={base_summary_i2['steps']} | fila_media={base_summary_i2['avg_queue_sum']:.2f} | "
            f"max_fila={base_summary_i2['max_queue_sum']} | throughput={base_summary_i2['throughput_total']} | "
            f"por_passo={base_summary_i2['throughput_per_step']:.2f}"
        )
    if not base_summary_i1 and not base_summary_i2:
        lines.append("- Sem dados.")
    lines.append("")
    lines.append("## Observacoes")
    lines.append("- Use este arquivo para comparar com o relatorio do MASPY.")
    lines.append("- Se desejar, rode o simulador MASPY e gere o relatorio consolidado.")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatorio comparativo gerado: {report}")


def main():
    run_dir = get_run_dir("baseline")
    write_run_readme(
        run_dir,
        {
            "type": "baseline",
            "seeds": [SEED, SEED + 1],
            "train_episodes": TRAIN_EPISODES,
            "horizon": HORIZON,
        },
    )
    q_table_i1 = train_sarsa(SEED)
    q_table_i2 = train_sarsa(SEED + 1)

    baseline_csv_i1 = run_dir / "metrics_baseline_I1.csv"
    baseline_csv_i2 = run_dir / "metrics_baseline_I2.csv"
    if PAIRED_ARRIVALS:
        evaluate(q_table_i1, baseline_csv_i1, ARRIVAL_SEED)
        evaluate(q_table_i2, baseline_csv_i2, ARRIVAL_SEED)
    else:
        evaluate(q_table_i1, baseline_csv_i1, ARRIVAL_SEED)
        evaluate(q_table_i2, baseline_csv_i2, ARRIVAL_SEED + 1)
    write_comparison_report(run_dir, baseline_csv_i1, baseline_csv_i2)
    print(f"CSV gerados: {baseline_csv_i1}, {baseline_csv_i2}")


if __name__ == "__main__":
    main()

