from __future__ import annotations

from src.benchmark.benchmark_suite import build_benchmark_suite
from src.benchmark.statistics import mean, summarize
from src.core.result import BenchmarkResult


def test_build_benchmark_suite_returns_multiple_experiments() -> None:
    suite = build_benchmark_suite()

    assert suite
    assert len(suite) > 1
    assert suite[0]["size"] == 256
    assert suite[0]["density_label"] == "low"


def test_summarize_computes_statistics() -> None:
    results = [
        BenchmarkResult(
            algorithm="A*",
            map_name="synthetic",
            width=256,
            height=256,
            density="low",
            scenario=0,
            execution_time_ms=3.0,
            preprocessing_time_ms=0.1,
            nodes_expanded=10,
            path_length=5.0,
            path_cost=5.0,
            portals=0,
            jump_points=0,
            abstract_graph_size=0,
            success=True,
        ),
        BenchmarkResult(
            algorithm="A*",
            map_name="synthetic",
            width=256,
            height=256,
            density="low",
            scenario=1,
            execution_time_ms=5.0,
            preprocessing_time_ms=0.2,
            nodes_expanded=20,
            path_length=6.0,
            path_cost=6.0,
            portals=0,
            jump_points=0,
            abstract_graph_size=0,
            success=True,
        ),
    ]

    summary = summarize(results)

    assert summary[0]["algorithm"] == "A*"
    assert summary[0]["mean_execution_time_ms"] == mean([3.0, 5.0])
    assert summary[0]["mean_nodes_expanded"] == mean([10, 20])


def test_run_experiment_with_real_map_scenarios() -> None:
    from pathlib import Path

    from src.benchmark.benchmark_suite import build_benchmark_suite
    from src.benchmark.experiment_runner import build_algorithms, run_experiment
    from src.io.map_loader import load_map, load_scenarios

    root = Path(__file__).resolve().parents[1] / "data" / "maps"
    if not root.exists():
        return

    suite = build_benchmark_suite(use_real_maps=True, repetitions=1)
    assert suite

    config = suite[0]
    grid = load_map(config["map_path"])
    scenarios: list = []
    for scen_path in config.get("scen_paths", []):
        scenarios.extend(load_scenarios(scen_path))
    scenarios = scenarios[:1]
    assert scenarios

    algorithms = build_algorithms(grid)
    results = run_experiment(config.get("map_path"), algorithms, scenarios, grid=grid, repetitions=1)

    assert results
    assert results[0].map_name == config["map_name"]
    assert results[0].success is True
    assert results[0].execution_time_ms > 0.0
