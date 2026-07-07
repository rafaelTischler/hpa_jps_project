"""CLI principal do projeto HPA-JPS."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.algorithms.astar import AStar
from src.algorithms.hpa_jps import HPAJPS
from src.algorithms.hpa_star import HPAStar
from src.algorithms.jps import JPS
from src.benchmark.experiment_runner import build_algorithms, run_experiment
from src.benchmark.metrics_writer import write_results
from src.io.map_loader import generate_synthetic_map, load_map, load_scenarios
from src.visualization.map_renderer import render_map
from src.visualization.plotter import plot_results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPA-JPS benchmark and visualization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Executa um caminho simples com um algoritmo")
    run_parser.add_argument("--map", default=None)
    run_parser.add_argument("--algorithm", default="astar")
    run_parser.add_argument("--start", default="0,0")
    run_parser.add_argument("--goal", default="50,50")

    experiment_parser = subparsers.add_parser("experiment", help="Executa um experimento resumido")
    experiment_parser.add_argument("--map", default=None)
    experiment_parser.add_argument("--scen", default=None, help="Arquivo .scen com pares start/goal reais (usado com --map)")
    experiment_parser.add_argument("--sizes", nargs="+", default=["256"])
    experiment_parser.add_argument("--densities", nargs="+", default=["low"])
    experiment_parser.add_argument("--map-count", type=int, default=1)
    experiment_parser.add_argument(
        "--num-scenarios",
        type=int,
        default=5,
        help="Quantos pares start/goal do arquivo .scen usar (apenas com --map real)",
    )

    plot_parser = subparsers.add_parser("plot", help="Gera gráficos a partir do CSV")
    plot_parser.add_argument("--input", default="results/raw/experiment_results.csv")

    render_parser = subparsers.add_parser("render", help="Renderiza um mapa")
    render_parser.add_argument("--map", default=None)
    render_parser.add_argument("--algorithm", default="hpajps")
    render_parser.add_argument("--start", default="0,0")
    render_parser.add_argument("--goal", default="10,10")

    return parser


def _parse_point(value: str) -> tuple[int, int]:
    x_str, y_str = value.split(",")
    return int(x_str), int(y_str)


def _build_algorithm(name: str, grid):
    name = name.lower()
    if name == "astar":
        return AStar(grid)
    if name == "jps":
        return JPS(grid)
    if name == "hpastar":
        return HPAStar(grid, cluster_size=10)
    if name == "hpajps":
        return HPAJPS(grid, cluster_size=10, hybrid=True)
    raise ValueError(f"unknown algorithm {name}")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        grid = load_map(args.map) if args.map else generate_synthetic_map(20, 20, 0.2, seed=42)
        algorithm = _build_algorithm(args.algorithm, grid)
        start = _parse_point(args.start)
        goal = _parse_point(args.goal)
        result = algorithm.search(start, goal)
        print(result)
        return 0

    if args.command == "experiment":
        if args.map and Path(args.map).exists():
            # Modo mapa real: carrega um .map real do movingai e (se fornecido) seus
            # pares start/goal reais de um .scen. --map-count é reinterpretado aqui
            # como "quantos pares start/goal amostrar do arquivo real", já que só
            # existe UM mapa (não faz sentido gerar múltiplos mapas sintéticos).
            grid = load_map(args.map)
            map_stem = Path(args.map).stem
            density_label = args.densities[0] if args.densities else "real"

            real_scenarios: list[Any]
            if args.scen and Path(args.scen).exists():
                all_scenarios = load_scenarios(args.scen)
                # Amostra espaçada ao longo do arquivo para pegar uma variedade de
                # distâncias curtas/longas, em vez de só as primeiras N linhas.
                count = max(1, args.num_scenarios)
                step = max(1, len(all_scenarios) // count)
                real_scenarios = all_scenarios[::step][:count]
                if not real_scenarios:
                    real_scenarios = all_scenarios[:count]
            else:
                # Sem .scen: usa canto-a-canto como único cenário (fallback simples).
                real_scenarios = [((0, 0), (grid.width - 1, grid.height - 1))]

            algorithms = build_algorithms(grid)
            results = run_experiment(None, algorithms, real_scenarios, grid=grid)
            for row in results:
                row["map_name"] = map_stem
                row["density"] = density_label
            output_path = write_results(results)
            grouped2: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in results:
                grouped2[row["algorithm"]].append(row)
            print(f"Wrote {len(results)} rows to {output_path} (mapa real: {map_stem}, {grid.width}x{grid.height})")
            for algorithm in sorted(grouped2):
                rows = grouped2[algorithm]
                avg_nodes = sum(row["nodes_expanded"] for row in rows) / max(1, len(rows))
                avg_exec = sum(row["execution_time_ms"] for row in rows) / max(1, len(rows))
                avg_prep = sum(row["preprocessing_time_ms"] for row in rows) / max(1, len(rows))
                print(f"{algorithm}: nodes_expanded={avg_nodes:.2f} execution_time_ms={avg_exec:.3f} preprocessing_time_ms={avg_prep:.3f}")
            return 0

        scenarios: list[dict[str, Any]] = []
        for size in args.sizes:
            width = height = int(size)
            for density in args.densities:
                density_value = {"low": 0.1, "medium": 0.25, "high": 0.4}.get(density.lower(), 0.25)
                for run_idx in range(args.map_count):
                    scenarios.append(
                        {
                            "width": width,
                            "height": height,
                            "density": density_value,
                            "start": (0, 0),
                            "goal": (width - 1, height - 1),
                        }
                    )
        first_size = int(args.sizes[0])
        first_density = {"low": 0.10, "medium": 0.25, "high": 0.40}.get(args.densities[0].lower(), 0.25)
        grid = generate_synthetic_map(first_size, first_size, first_density, seed=42)
        algorithms = build_algorithms(grid)
        results = run_experiment(args.map, algorithms, scenarios, grid=None)
        output_path = write_results(results)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            grouped[row["algorithm"]].append(row)
        print(f"Wrote {len(results)} rows to {output_path}")
        for algorithm in sorted(grouped):
            rows = grouped[algorithm]
            avg_nodes = sum(row["nodes_expanded"] for row in rows) / max(1, len(rows))
            avg_exec = sum(row["execution_time_ms"] for row in rows) / max(1, len(rows))
            avg_graph = sum(row["abstract_graph_size"] for row in rows) / max(1, len(rows))
            print(f"{algorithm}: nodes_expanded={avg_nodes:.2f} execution_time_ms={avg_exec:.3f} abstract_graph_size={avg_graph:.2f}")
        return 0

    if args.command == "plot":
        files = plot_results(args.input)
        print("\n".join(files))
        return 0

    if args.command == "render":
        grid = load_map(args.map) if args.map else generate_synthetic_map(20, 20, 0.2, seed=42)
        start = _parse_point(args.start)
        goal = _parse_point(args.goal)
        algorithm = _build_algorithm(args.algorithm, grid)
        result = algorithm.search(start, goal)
        output_path = render_map(grid, path=result.path, title=args.algorithm.upper())
        print(output_path)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
