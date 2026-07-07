"""Geração de gráficos a partir dos resultados de benchmark."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_results(input_path: str | Path, output_dir: str | Path | None = None) -> list[str]:
    """Lê o CSV de resultados e gera os gráficos solicitados em ``results/plots``."""
    input_file = Path(input_path)
    output_dir = Path(output_dir or "results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_results(input_file)
    if not rows:
        raise ValueError(f"No benchmark data found in {input_file}")

    generated: list[str] = []
    generated.append(_plot_execution_time(rows, output_dir))
    generated.append(_plot_nodes_expanded(rows, output_dir))
    generated.append(_plot_abstract_graph(rows, output_dir))
    generated.append(_plot_path_quality(rows, output_dir))
    generated.append(_write_summary_table(rows, output_dir))
    return generated


def _read_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "map_name": row["map_name"],
                "size": row["size"],
                "density": row["density"],
                "algorithm": row["algorithm"],
                "run_id": int(row["run_id"]),
                "execution_time_ms": float(row["execution_time_ms"]),
                "nodes_expanded": int(float(row["nodes_expanded"])),
                "path_length": float(row["path_length"]),
                "path_cost": float(row["path_cost"]),
                "num_portals": int(float(row["num_portals"])),
                "num_jump_points": int(float(row["num_jump_points"])),
                "abstract_graph_size": int(float(row["abstract_graph_size"])),
                "success": row["success"].lower() == "true",
            }
            for row in reader
        ]


def _plot_execution_time(rows: list[dict[str, Any]], output_dir: Path) -> str:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["size"], row["algorithm"]), []).append(row["execution_time_ms"])

    sizes = sorted({row["size"] for row in rows})
    algorithms = sorted({row["algorithm"] for row in rows})
    values: dict[str, list[float]] = {algo: [] for algo in algorithms}
    for size in sizes:
        for algo in algorithms:
            timings = grouped.get((size, algo), [])
            values[algo].append(sum(timings) / len(timings) if timings else math.nan)

    x = list(range(len(sizes)))
    width = 0.8 / max(1, len(algorithms))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, algo in enumerate(algorithms):
        ax.bar([xi + (idx - (len(algorithms) - 1) / 2) * width for xi in x], values[algo], width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Tempo médio (ms)")
    ax.set_title("Tempo médio por algoritmo por tamanho")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "execution_time_by_algorithm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_nodes_expanded(rows: list[dict[str, Any]], output_dir: Path) -> str:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row["nodes_expanded"])

    algorithms = sorted(grouped)
    averages = [sum(grouped[algo]) / len(grouped[algo]) for algo in algorithms]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(algorithms, averages)
    ax.set_ylabel("Nós expandidos (média)")
    ax.set_title("Nós expandidos por algoritmo")
    fig.tight_layout()
    path = output_dir / "nodes_expanded_by_algorithm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_abstract_graph(rows: list[dict[str, Any]], output_dir: Path) -> str:
    densities = sorted({row["density"] for row in rows if row["algorithm"] in {"HPAStar", "HPAJPS"}})
    values: dict[str, list[float]] = {"HPAStar": [], "HPAJPS": []}
    for density in densities:
        for algo in values:
            subset = [row["abstract_graph_size"] for row in rows if row["algorithm"] == algo and row["density"] == density]
            values[algo].append(sum(subset) / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(densities)))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], values["HPAStar"], width=width, label="HPAStar")
    ax.bar([xi + width / 2 for xi in x], values["HPAJPS"], width=width, label="HPAJPS")
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.set_ylabel("Tamanho do grafo abstrato")
    ax.set_title("Grafo abstrato HPA* vs HPA-JPS por densidade")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "abstract_graph_size_by_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_path_quality(rows: list[dict[str, Any]], output_dir: Path) -> str:
    algorithms = sorted({row["algorithm"] for row in rows})
    costs = []
    lengths = []
    for algo in algorithms:
        subset = [row for row in rows if row["algorithm"] == algo and row["success"]]
        costs.append(sum(row["path_cost"] for row in subset) / len(subset) if subset else 0.0)
        lengths.append(sum(row["path_length"] for row in subset) / len(subset) if subset else 0.0)
    x = list(range(len(algorithms)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([xi - width / 2 for xi in x], costs, width=width, label="custo médio")
    ax.bar([xi + width / 2 for xi in x], lengths, width=width, label="comprimento médio")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("Métrica de qualidade")
    ax.set_title("Qualidade do caminho entre algoritmos")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "path_quality_by_algorithm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _write_summary_table(rows: list[dict[str, Any]], output_dir: Path) -> str:
    summary_rows = []
    algorithms = sorted({row["algorithm"] for row in rows})
    for algo in algorithms:
        success = [row for row in rows if row["algorithm"] == algo and row["success"]]
        summary_rows.append(
            {
                "algorithm": algo,
                "mean_execution_time_ms": round(sum(row["execution_time_ms"] for row in rows if row["algorithm"] == algo) / max(1, len([r for r in rows if r["algorithm"] == algo])), 3),
                "mean_nodes_expanded": round(sum(row["nodes_expanded"] for row in rows if row["algorithm"] == algo) / max(1, len([r for r in rows if r["algorithm"] == algo])), 3),
                "success_rate": round(len(success) / max(1, len([r for r in rows if r["algorithm"] == algo])), 3),
            }
        )

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "mean_execution_time_ms", "mean_nodes_expanded", "success_rate"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = output_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# Resumo do benchmark\n\n")
        handle.write("| algoritmo | tempo médio (ms) | nós expandidos | taxa de sucesso |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(f"| {row['algorithm']} | {row['mean_execution_time_ms']} | {row['mean_nodes_expanded']} | {row['success_rate']} |\n")

    return str(summary_csv)
