"""Geração automática de gráficos para os resultados do benchmark."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def generate_plots(input_path: str | Path, output_dir: str | Path | None = None) -> list[str]:
    """Gera os gráficos do benchmark a partir do CSV bruto."""
    input_file = Path(input_path)
    output_dir = Path(output_dir or "results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_file)
    if not rows:
        raise ValueError(f"No benchmark data found in {input_file}")

    generated = []
    generated.append(_plot_execution_time(rows, output_dir))
    generated.append(_plot_nodes(rows, output_dir))
    generated.append(_plot_path_length(rows, output_dir))
    generated.append(_plot_graph_size(rows, output_dir))
    generated.append(_plot_jump_points(rows, output_dir))
    generated.append(_plot_portals(rows, output_dir))
    return generated


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "algorithm": row["algorithm"],
                "map_name": row["map_name"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "density": row["density"],
                "scenario": int(row["scenario"]),
                "execution_time_ms": float(row["execution_time_ms"]),
                "nodes_expanded": int(float(row["nodes_expanded"])),
                "path_length": float(row["path_length"]),
                "path_cost": float(row["path_cost"]),
                "portals": int(float(row["portals"])),
                "jump_points": int(float(row["jump_points"])),
                "abstract_graph_size": int(float(row["abstract_graph_size"])),
                "success": row["success"].lower() == "true",
            }
            for row in reader
        ]


def _plot_execution_time(rows: list[dict[str, Any]], output_dir: Path) -> str:
    algorithms = sorted({row["algorithm"] for row in rows})
    sizes = sorted({f"{row['width']}x{row['height']}" for row in rows})
    values: dict[str, list[float]] = {algo: [] for algo in algorithms}
    for size in sizes:
        for algo in algorithms:
            subset = [row for row in rows if f"{row['width']}x{row['height']}" == size and row["algorithm"] == algo]
            values[algo].append(sum(r["execution_time_ms"] for r in subset) / len(subset) if subset else math.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(sizes)))
    width = 0.8 / max(1, len(algorithms))
    for idx, algo in enumerate(algorithms):
        ax.bar([xi + (idx - (len(algorithms) - 1) / 2) * width for xi in x], values[algo], width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Tempo médio (ms)")
    ax.set_title("Tempo por tamanho")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "tempo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_nodes(rows: list[dict[str, Any]], output_dir: Path) -> str:
    algorithms = sorted({row["algorithm"] for row in rows})
    averages = []
    for algo in algorithms:
        subset = [row for row in rows if row["algorithm"] == algo and row["success"]]
        averages.append(sum(r["nodes_expanded"] for r in subset) / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(algorithms, averages)
    ax.set_ylabel("Nós expandidos")
    ax.set_title("Nós expandidos por algoritmo")
    fig.tight_layout()
    path = output_dir / "nos.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_path_length(rows: list[dict[str, Any]], output_dir: Path) -> str:
    algorithms = sorted({row["algorithm"] for row in rows})
    values = []
    for algo in algorithms:
        subset = [row for row in rows if row["algorithm"] == algo and row["success"]]
        values.append(sum(r["path_length"] for r in subset) / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(algorithms, values)
    ax.set_ylabel("Comprimento médio")
    ax.set_title("Comprimento do caminho")
    fig.tight_layout()
    path = output_dir / "comprimento.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_graph_size(rows: list[dict[str, Any]], output_dir: Path) -> str:
    algorithms = [algo for algo in ["HPAStar", "HPAJPS"] if any(row["algorithm"] == algo for row in rows)]
    densities = sorted({row["density"] for row in rows if row["algorithm"] in algorithms})
    values: dict[str, list[float]] = {algo: [] for algo in algorithms}
    for density in densities:
        for algo in algorithms:
            subset = [row for row in rows if row["algorithm"] == algo and row["density"] == density]
            values[algo].append(sum(r["abstract_graph_size"] for r in subset) / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(densities)))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], values.get("HPAStar", []), width=width, label="HPAStar")
    ax.bar([xi + width / 2 for xi in x], values.get("HPAJPS", []), width=width, label="HPAJPS")
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.set_ylabel("Grafo abstrato")
    ax.set_title("Tamanho do grafo abstrato")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "grafo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_jump_points(rows: list[dict[str, Any]], output_dir: Path) -> str:
    subset = [row for row in rows if row["algorithm"] == "HPAJPS"]
    if not subset:
        return ""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["jump_points"] for row in subset], bins=10)
    ax.set_xlabel("Jump points")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de jump points")
    fig.tight_layout()
    path = output_dir / "jump_points.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_portals(rows: list[dict[str, Any]], output_dir: Path) -> str:
    subset = [row for row in rows if row["algorithm"] == "HPAStar"]
    if not subset:
        return ""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["portals"] for row in subset], bins=10)
    ax.set_xlabel("Portais")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de portais")
    fig.tight_layout()
    path = output_dir / "portais.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
