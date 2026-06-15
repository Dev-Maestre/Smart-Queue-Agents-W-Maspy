import csv
import statistics
from pathlib import Path


REPORT_FILE = "relatorio_consolidado.md"
INPUT_FILES = ["metrics_I1.csv", "metrics_I2.csv"]


def _read_metrics(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("time"):
                continue
            rows.append(
                {
                    "time": int(row["time"]),
                    "phase": row["phase"],
                    "queue_sum": int(row["queue_sum"]),
                    "q_n": int(row["q_n"]),
                    "q_s": int(row["q_s"]),
                    "q_e": int(row["q_e"]),
                    "q_w": int(row["q_w"]),
                    "served_n": int(row["served_n"]),
                    "served_s": int(row["served_s"]),
                    "served_e": int(row["served_e"]),
                    "served_w": int(row["served_w"]),
                }
            )
    return rows


def _summarize(rows):
    if not rows:
        return {}

    steps = len(rows)
    phase_counts = {}
    queue_sum_values = [r["queue_sum"] for r in rows]
    qn = [r["q_n"] for r in rows]
    qs = [r["q_s"] for r in rows]
    qe = [r["q_e"] for r in rows]
    qw = [r["q_w"] for r in rows]
    for r in rows:
        phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1

    max_queue_cap = max(qn + qs + qe + qw)
    saturated_steps = sum(
        1 for r in rows if max(r["q_n"], r["q_s"], r["q_e"], r["q_w"]) >= max_queue_cap
    )

    throughput = {
        "N": sum(r["served_n"] for r in rows),
        "S": sum(r["served_s"] for r in rows),
        "E": sum(r["served_e"] for r in rows),
        "W": sum(r["served_w"] for r in rows),
    }

    worst_steps = sorted(rows, key=lambda r: r["queue_sum"], reverse=True)[:5]

    return {
        "steps": steps,
        "phase_counts": phase_counts,
        "avg_queue_sum": statistics.mean(queue_sum_values),
        "max_queue_sum": max(queue_sum_values),
        "std_queue_sum": statistics.pstdev(queue_sum_values),
        "avg_q_n": statistics.mean(qn),
        "avg_q_s": statistics.mean(qs),
        "avg_q_e": statistics.mean(qe),
        "avg_q_w": statistics.mean(qw),
        "throughput": throughput,
        "throughput_total": sum(throughput.values()),
        "throughput_per_step": sum(throughput.values()) / steps,
        "saturated_steps": saturated_steps,
        "saturated_pct": (saturated_steps / steps) * 100.0,
        "worst_steps": worst_steps,
        "queue_cap_obs": max_queue_cap,
    }


def _format_phase_counts(phase_counts, steps):
    parts = []
    for phase, count in sorted(phase_counts.items(), key=lambda x: (-x[1], x[0])):
        parts.append(f"{phase}: {count} ({(count/steps)*100:.1f}%)")
    return ", ".join(parts)


def _render_report(data):
    lines = []
    lines.append("# Relatorio consolidado - Semaforos Inteligentes")
    lines.append("")
    lines.append("## Resumo")
    lines.append(
        "- Este relatorio consolida as metricas de duas intersecoes (I1 e I2) a partir dos arquivos CSV gerados pelo simulador."
    )
    lines.append("- As metricas permitem avaliar fluxo, filas e estabilidade do controle.")
    lines.append("")

    for name, summary in data.items():
        if not summary:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Passos simulados: {summary['steps']}")
        lines.append(
            f"- Fase ativa (distribuicao): { _format_phase_counts(summary['phase_counts'], summary['steps']) }"
        )
        lines.append(
            f"- Fila media total: {summary['avg_queue_sum']:.2f} | Max: {summary['max_queue_sum']} | Desvio-padrao: {summary['std_queue_sum']:.2f}"
        )
        lines.append(
            f"- Media por direcao: N={summary['avg_q_n']:.2f}, S={summary['avg_q_s']:.2f}, E={summary['avg_q_e']:.2f}, W={summary['avg_q_w']:.2f}"
        )
        lines.append(
            f"- Throughput total: {summary['throughput_total']} | por passo: {summary['throughput_per_step']:.2f}"
        )
        lines.append(
            f"- Throughput por direcao: N={summary['throughput']['N']}, S={summary['throughput']['S']}, E={summary['throughput']['E']}, W={summary['throughput']['W']}"
        )
        lines.append(
            f"- Saturacao (fila no max observado={summary['queue_cap_obs']}): {summary['saturated_steps']} passos ({summary['saturated_pct']:.1f}%)"
        )
        lines.append("")
        lines.append("### Piores momentos (maior fila total)")
        for r in summary["worst_steps"]:
            lines.append(
                f"- t={r['time']}: phase={r['phase']} | fila_total={r['queue_sum']} | N={r['q_n']} S={r['q_s']} E={r['q_e']} W={r['q_w']}"
            )
        lines.append("")

    lines.append("## Observacoes")
    lines.append(
        "- Quanto menor a fila media total e maior o throughput por passo, melhor o desempenho do controle."
    )
    lines.append(
        "- A distribuicao das fases mostra se o sistema esta equilibrado entre eixos NS/EW e fases individuais."
    )
    lines.append(
        "- A saturacao indica quantas vezes alguma direcao ficou no limite de capacidade observado."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    base = Path(__file__).parent
    data = {}
    for fname in INPUT_FILES:
        path = base / fname
        rows = _read_metrics(path) if path.exists() else []
        key = path.stem.upper()
        data[key] = _summarize(rows)
    report = _render_report(data)
    (base / REPORT_FILE).write_text(report, encoding="utf-8")
    print(f"Relatorio gerado em: {base / REPORT_FILE}")


if __name__ == "__main__":
    main()

