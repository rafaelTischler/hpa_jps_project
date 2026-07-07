"""Escrita de métricas de benchmark em CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from src.benchmark.statistics import summarize
from src.core.result import BenchmarkResult


def write_results(
    results: Sequence[dict[str, Any]] | Sequence[BenchmarkResult],
    output_path: str | Path = "results/raw/experiment_results.csv",
) -> str:
    """Escreve os resultados em CSV com as colunas exigidas pelo SPEC."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "map_name",
        "size",
        "density",
        "algorithm",
        "run_id",
        "execution_time_ms",
        "nodes_expanded",
        "path_length",
        "path_cost",
        "num_portals",
        "num_jump_points",
        "abstract_graph_size",
        "preprocessing_time_ms",
        "preprocessing_nodes_expanded",
        "success",
    ]

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if isinstance(row, BenchmarkResult):
                writer.writerow(
                    {
                        "map_name": row.map_name,
                        "size": f"{row.width}x{row.height}",
                        "density": row.density,
                        "algorithm": row.algorithm,
                        "run_id": row.scenario,
                        "execution_time_ms": row.execution_time_ms,
                        "nodes_expanded": row.nodes_expanded,
                        "path_length": row.path_length,
                        "path_cost": row.path_cost,
                        "num_portals": row.portals,
                        "num_jump_points": row.jump_points,
                        "abstract_graph_size": row.abstract_graph_size,
                        "preprocessing_time_ms": row.preprocessing_time_ms,
                        "preprocessing_nodes_expanded": 0,
                        "success": row.success,
                    }
                )
            else:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    return str(output)


def write_benchmark_outputs(results: Sequence[BenchmarkResult], raw_path: str | Path = "results/raw/raw_results.csv") -> tuple[str, str]:
    """Escreve os CSVs brutos e resumidos do benchmark."""
    raw_output = Path(raw_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    summary_output = raw_output.parent / "summary.csv"
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "map_name",
        "width",
        "height",
        "density",
        "scenario",
        "execution_time_ms",
        "preprocessing_time_ms",
        "nodes_expanded",
        "path_length",
        "path_cost",
        "portals",
        "jump_points",
        "abstract_graph_size",
        "success",
    ]

    with raw_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "algorithm": row.algorithm,
                    "map_name": row.map_name,
                    "width": row.width,
                    "height": row.height,
                    "density": row.density,
                    "scenario": row.scenario,
                    "execution_time_ms": row.execution_time_ms,
                    "preprocessing_time_ms": row.preprocessing_time_ms,
                    "nodes_expanded": row.nodes_expanded,
                    "path_length": row.path_length,
                    "path_cost": row.path_cost,
                    "portals": row.portals,
                    "jump_points": row.jump_points,
                    "abstract_graph_size": row.abstract_graph_size,
                    "success": row.success,
                }
            )

    summary_rows = summarize(results)
    summary_fieldnames = [
        "algorithm",
        "mean_execution_time_ms",
        "median_execution_time_ms",
        "std_execution_time_ms",
        "min_execution_time_ms",
        "max_execution_time_ms",
        "mean_nodes_expanded",
        "median_nodes_expanded",
        "std_nodes_expanded",
        "min_nodes_expanded",
        "max_nodes_expanded",
        "success_rate",
    ]
    with summary_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in summary_fieldnames})

    return str(raw_output), str(summary_output)
