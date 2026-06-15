import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from run_utils import get_latest_run_dir

from config import ADAPT_SHOCK_STEP


BASE_DIR = Path(__file__).parent
LABEL_MASPY_I1 = "MASPY I1"
LABEL_MASPY_I2 = "MASPY I2"
LABEL_BASE_I1 = "Baseline I1"
LABEL_BASE_I2 = "Baseline I2"


def _safe_yerr(means, lows, highs):
    # Matplotlib exige yerr >= 0; protege contra summaries inconsistentes.
    low_err = [max(0.0, m - l) for m, l in zip(means, lows)]
    high_err = [max(0.0, h - m) for m, h in zip(means, highs)]
    return [low_err, high_err]


def _is_episode0(row):
    ep = row.get("episode")
    return ep in (None, "", 0, "0")


def read_timeseries(csv_path, key):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [int(row[key]) for row in reader if row.get(key) and _is_episode0(row)]


def read_time_and_series(csv_path, key):
    if not csv_path.exists():
        return [], []
    times = []
    values = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("time") and row.get(key) and _is_episode0(row):
                times.append(int(row["time"]))
                values.append(int(row[key]))
    return times, values


def read_time_and_throughput(csv_path):
    if not csv_path.exists():
        return [], []
    times = []
    values = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_episode0(row):
                continue
            if not row.get("time"):
                continue
            served_n = int(row.get("served_n", 0) or 0)
            served_s = int(row.get("served_s", 0) or 0)
            served_e = int(row.get("served_e", 0) or 0)
            served_w = int(row.get("served_w", 0) or 0)
            times.append(int(row["time"]))
            values.append(served_n + served_s + served_e + served_w)
    return times, values


def read_summary(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_queue_timeseries(run_dir: Path):
    plt.figure(figsize=(10, 5))
    t1_x, t1 = read_time_and_series(run_dir / "metrics_I1.csv", "queue_sum")
    t2_x, t2 = read_time_and_series(run_dir / "metrics_I2.csv", "queue_sum")
    if t1:
        plt.plot(t1_x, t1, label=LABEL_MASPY_I1)
    if t2:
        plt.plot(t2_x, t2, label=LABEL_MASPY_I2)

    b1_x, b1 = read_time_and_series(run_dir / "metrics_baseline_I1.csv", "queue_sum")
    b2_x, b2 = read_time_and_series(run_dir / "metrics_baseline_I2.csv", "queue_sum")
    if b1:
        plt.plot(b1_x, b1, label=LABEL_BASE_I1, linestyle="--")
    if b2:
        plt.plot(b2_x, b2, label=LABEL_BASE_I2, linestyle="--")

    plt.title("Queue sum por tempo")
    plt.xlabel("Passo")
    plt.ylabel("Soma das filas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_queue_timeseries.png", dpi=150)
    plt.close()

    # Plot alinhado pelo menor horizonte para comparacao visual direta
    min_len = min([len(x) for x in [t1, t2, b1, b2] if x] or [0])
    if min_len > 0:
        plt.figure(figsize=(10, 5))
        if t1:
            plt.plot(range(1, min_len + 1), t1[:min_len], label=LABEL_MASPY_I1)
        if t2:
            plt.plot(range(1, min_len + 1), t2[:min_len], label=LABEL_MASPY_I2)
        if b1:
            plt.plot(range(1, min_len + 1), b1[:min_len], label=LABEL_BASE_I1, linestyle="--")
        if b2:
            plt.plot(range(1, min_len + 1), b2[:min_len], label=LABEL_BASE_I2, linestyle="--")
        plt.title("Queue sum por tempo (alinhado)")
        plt.xlabel("Passo")
        plt.ylabel("Soma das filas")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "fig_queue_timeseries_aligned.png", dpi=150)
        plt.close()


def plot_throughput_timeseries(run_dir: Path):
    plt.figure(figsize=(10, 5))
    t1_x, t1 = read_time_and_throughput(run_dir / "metrics_I1.csv")
    t2_x, t2 = read_time_and_throughput(run_dir / "metrics_I2.csv")
    if t1:
        plt.plot(t1_x, t1, label=LABEL_MASPY_I1)
    if t2:
        plt.plot(t2_x, t2, label=LABEL_MASPY_I2)

    b1_x, b1 = read_time_and_throughput(run_dir / "metrics_baseline_I1.csv")
    b2_x, b2 = read_time_and_throughput(run_dir / "metrics_baseline_I2.csv")
    if b1:
        plt.plot(b1_x, b1, label=LABEL_BASE_I1, linestyle="--")
    if b2:
        plt.plot(b2_x, b2, label=LABEL_BASE_I2, linestyle="--")

    plt.title("Throughput por tempo")
    plt.xlabel("Passo")
    plt.ylabel("Throughput (servidos por passo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_throughput_timeseries.png", dpi=150)
    plt.close()

    min_len = min([len(x) for x in [t1, t2, b1, b2] if x] or [0])
    if min_len > 0:
        plt.figure(figsize=(10, 5))
        if t1:
            plt.plot(range(1, min_len + 1), t1[:min_len], label=LABEL_MASPY_I1)
        if t2:
            plt.plot(range(1, min_len + 1), t2[:min_len], label=LABEL_MASPY_I2)
        if b1:
            plt.plot(range(1, min_len + 1), b1[:min_len], label=LABEL_BASE_I1, linestyle="--")
        if b2:
            plt.plot(range(1, min_len + 1), b2[:min_len], label=LABEL_BASE_I2, linestyle="--")
        plt.title("Throughput por tempo (alinhado)")
        plt.xlabel("Passo")
        plt.ylabel("Throughput (servidos por passo)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "fig_throughput_timeseries_aligned.png", dpi=150)
        plt.close()


def plot_summary_bars(run_dir: Path):
    summary = read_summary(run_dir / "experimentos_summary.csv")
    if not summary:
        return
    maspy_q = [float(r["maspy_avg_queue"]) for r in summary]
    base_q = [float(r["baseline_avg_queue"]) for r in summary]
    maspy_t = [float(r["maspy_throughput"]) for r in summary]
    base_t = [float(r["baseline_throughput"]) for r in summary]

    def mean_ci(vals):
        if not vals:
            return (0.0, 0.0)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        sd = var ** 0.5
        half = 1.96 * sd / (len(vals) ** 0.5)
        return mean, half

    mq, mq_ci = mean_ci(maspy_q)
    bq, bq_ci = mean_ci(base_q)
    mt, mt_ci = mean_ci(maspy_t)
    bt, bt_ci = mean_ci(base_t)

    plt.figure(figsize=(10, 6))
    bars_q = plt.bar(["MASPY", "Baseline"], [mq, bq], yerr=[mq_ci, bq_ci], capsize=8)
    plt.title("Fila media total (IC95%)")
    plt.ylabel("Fila media")
    y_max_q = max(mq + mq_ci, bq + bq_ci, 1e-9)
    plt.ylim(0, y_max_q * 1.20)

    for bar, val in zip(bars_q, [mq, bq]):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, val + y_max_q * 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    if bq > 0:
        # Para fila, menor eh melhor: percentual positivo significa reducao.
        diff_pct_q = ((bq - mq) / bq) * 100.0
        pct_text_q = f"Reducao de fila MASPY vs Baseline: {diff_pct_q:.1f}%"
    else:
        pct_text_q = "Reducao de fila MASPY vs Baseline: n/a (baseline=0)"
    plt.text(0.5, y_max_q * 1.11, pct_text_q, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(run_dir / "fig_queue_mean_ci.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(["MASPY", "Baseline"], [mt, bt], yerr=[mt_ci, bt_ci], capsize=6)
    plt.title("Throughput por passo (IC95%)")
    plt.ylabel("Throughput")
    y_max = max(mt + mt_ci, bt + bt_ci, 1e-9)
    plt.ylim(0, y_max * 1.20)

    for bar, val in zip(bars, [mt, bt]):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, val + y_max * 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    if bt > 0:
        diff_pct = ((mt - bt) / bt) * 100.0
        pct_text = f"Diferenca MASPY vs Baseline: {diff_pct:.1f}%"
    else:
        pct_text = "Diferenca MASPY vs Baseline: n/a (baseline=0)"
    plt.text(0.5, y_max * 1.11, pct_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(run_dir / "fig_throughput_mean_ci.png", dpi=150)
    plt.close()


def plot_reward_curves(run_dir: Path):
    rows = read_rows(run_dir / "training_rewards_comparison.csv")
    if not rows:
        return
    episodes = [int(r["episode"]) for r in rows]
    m_reward = [float(r["maspy_reward_mean"]) for r in rows]
    b_reward = [float(r["baseline_reward_mean"]) for r in rows]
    m_cum = [float(r["maspy_cumulative_reward"]) for r in rows]
    b_cum = [float(r["baseline_cumulative_reward"]) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, m_reward, label="MASPY")
    plt.plot(episodes, b_reward, label="Baseline", linestyle="--")
    plt.title("Recompensa media por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_reward_per_episode.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, m_cum, label="MASPY")
    plt.plot(episodes, b_cum, label="Baseline", linestyle="--")
    plt.title("Recompensa cumulativa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa cumulativa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_cumulative_reward.png", dpi=150)
    plt.close()


def plot_convergence_bars(run_dir: Path):
    rows = read_rows(run_dir / "convergence_summary.csv")
    if not rows:
        return
    labels = [r["algorithm"] for r in rows]
    means = [float(r["mean_convergence_episode"]) for r in rows]
    lows = [float(r["ci95_low"]) for r in rows]
    highs = [float(r["ci95_high"]) for r in rows]
    errs = _safe_yerr(means, lows, highs)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=errs, capsize=8)
    plt.title("Velocidade de convergencia (episodios)")
    plt.ylabel("Episodio de convergencia")
    y_max = max((m + e[1] for m, e in zip(means, zip(errs[0], errs[1]))), default=1.0)
    plt.ylim(0, y_max * 1.20)

    for bar, val in zip(bars, means):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, val + y_max * 0.03, f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    if len(means) >= 2 and means[1] > 0:
        diff_pct = ((means[1] - means[0]) / means[1]) * 100.0
        pct_text = f"Diferenca MASPY vs Baseline: {diff_pct:.1f}%"
    else:
        pct_text = "Diferenca MASPY vs Baseline: n/a"
    plt.text(0.5, y_max * 1.11, pct_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(run_dir / "fig_convergence_rate.png", dpi=150)
    plt.close()


def plot_adaptation_shock(run_dir: Path):
    rows = read_rows(run_dir / "adaptation_shock_timeseries.csv")
    if not rows:
        return

    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        algo = r["algorithm"]
        step = int(r["step"])
        grouped[algo][step].append(float(r["throughput"]))

    def mean_by_step(step_map):
        xs = sorted(step_map.keys())
        ys = [sum(step_map[x]) / len(step_map[x]) for x in xs]
        return xs, ys

    plt.figure(figsize=(10, 5))
    for algo, step_map in grouped.items():
        xs, ys = mean_by_step(step_map)
        style = "-" if algo == "MASPY" else "--"
        plt.plot(xs, ys, label=algo, linestyle=style)
    plt.axvline(ADAPT_SHOCK_STEP, color="red", linestyle=":", label="Inicio choque")
    plt.title("Adaptacao ao choque de demanda (throughput medio)")
    plt.xlabel("Passo")
    plt.ylabel("Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_adaptation_shock_throughput.png", dpi=150)
    plt.close()

    grouped_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        algo = r["algorithm"]
        step = int(r["step"])
        grouped_q[algo][step].append(float(r["queue_sum"]))

    plt.figure(figsize=(10, 5))
    for algo, step_map in grouped_q.items():
        xs, ys = mean_by_step(step_map)
        style = "-" if algo == "MASPY" else "--"
        plt.plot(xs, ys, label=algo, linestyle=style)
    plt.axvline(ADAPT_SHOCK_STEP, color="red", linestyle=":", label="Inicio choque")
    plt.title("Adaptacao ao choque de demanda (fila media)")
    plt.xlabel("Passo")
    plt.ylabel("Soma das filas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "fig_adaptation_shock_queue.png", dpi=150)
    plt.close()


def plot_recovery_bars(run_dir: Path):
    rows = read_rows(run_dir / "adaptation_recovery_summary.csv")
    if not rows:
        return
    labels = [r["algorithm"] for r in rows]
    means = [float(r["mean_recovery_steps"]) for r in rows]
    lows = [float(r["ci95_low"]) for r in rows]
    highs = [float(r["ci95_high"]) for r in rows]
    errs = _safe_yerr(means, lows, highs)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=errs, capsize=8)
    plt.title("Tempo de recuperacao pos-choque")
    plt.ylabel("Passos para recuperar throughput")
    y_max = max((m + e[1] for m, e in zip(means, zip(errs[0], errs[1]))), default=1.0)
    plt.ylim(0, y_max * 1.20)

    for bar, val in zip(bars, means):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, val + y_max * 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    if len(means) >= 2 and means[1] > 0:
        diff_pct = ((means[1] - means[0]) / means[1]) * 100.0
        pct_text = f"Diferenca MASPY vs Baseline: {diff_pct:.1f}%"
    else:
        pct_text = "Diferenca MASPY vs Baseline: n/a"
    plt.text(0.5, y_max * 1.11, pct_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(run_dir / "fig_adaptation_recovery.png", dpi=150)
    plt.close()


def main():
    run_dir = get_latest_run_dir() or BASE_DIR
    plot_queue_timeseries(run_dir)
    plot_throughput_timeseries(run_dir)
    plot_summary_bars(run_dir)
    plot_reward_curves(run_dir)
    plot_convergence_bars(run_dir)
    plot_adaptation_shock(run_dir)
    plot_recovery_bars(run_dir)
    print("Graficos gerados:")
    print(f"- {run_dir / 'fig_queue_timeseries.png'}")
    print(f"- {run_dir / 'fig_queue_timeseries_aligned.png'}")
    print(f"- {run_dir / 'fig_throughput_timeseries.png'}")
    print(f"- {run_dir / 'fig_throughput_timeseries_aligned.png'}")
    print(f"- {run_dir / 'fig_queue_mean_ci.png'}")
    print(f"- {run_dir / 'fig_throughput_mean_ci.png'}")
    print(f"- {run_dir / 'fig_reward_per_episode.png'}")
    print(f"- {run_dir / 'fig_cumulative_reward.png'}")
    print(f"- {run_dir / 'fig_convergence_rate.png'}")
    print(f"- {run_dir / 'fig_adaptation_shock_throughput.png'}")
    print(f"- {run_dir / 'fig_adaptation_shock_queue.png'}")
    print(f"- {run_dir / 'fig_adaptation_recovery.png'}")


if __name__ == "__main__":
    main()

