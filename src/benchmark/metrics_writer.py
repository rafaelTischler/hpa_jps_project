"""Escrita de métricas de benchmark em CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def write_results(
    results: Sequence[dict[str, Any]],
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
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return str(output)
