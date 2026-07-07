"""Funções utilitárias para calcular estatísticas de benchmark."""

from __future__ import annotations

import math
from statistics import mean as _mean, median as _median, pstdev as _pstdev
from typing import Sequence

from src.core.result import BenchmarkResult


def mean(values: Sequence[float]) -> float:
    return _mean(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return _median(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    return _pstdev(values) if len(values) > 1 else 0.0


def minimum(values: Sequence[float]) -> float:
    return min(values) if values else 0.0


def maximum(values: Sequence[float]) -> float:
    return max(values) if values else 0.0


def summarize(results: Sequence[BenchmarkResult]) -> list[dict[str, float | int | str | bool]]:
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.algorithm, []).append(result)

    summaries: list[dict[str, float | int | str | bool]] = []
    for algorithm, algorithm_results in sorted(grouped.items()):
        execution_times = [row.execution_time_ms for row in algorithm_results]
        nodes_expanded = [float(row.nodes_expanded) for row in algorithm_results]
        summaries.append(
            {
                "algorithm": algorithm,
                "mean_execution_time_ms": round(mean(execution_times), 6),
                "median_execution_time_ms": round(median(execution_times), 6),
                "std_execution_time_ms": round(std(execution_times), 6),
                "min_execution_time_ms": round(minimum(execution_times), 6),
                "max_execution_time_ms": round(maximum(execution_times), 6),
                "mean_nodes_expanded": round(mean(nodes_expanded), 6),
                "median_nodes_expanded": round(median(nodes_expanded), 6),
                "std_nodes_expanded": round(std(nodes_expanded), 6),
                "min_nodes_expanded": round(minimum(nodes_expanded), 6),
                "max_nodes_expanded": round(maximum(nodes_expanded), 6),
                "success_rate": round(sum(1 for row in algorithm_results if row.success) / max(1, len(algorithm_results)), 6),
            }
        )
    return summaries
