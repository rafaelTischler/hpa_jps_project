"""Execução de experimentos de benchmark para os algoritmos de pathfinding."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence
import os
import multiprocessing

from src.algorithms.astar import AStar
from src.algorithms.base_pathfinder import PathfindingAlgorithm
from src.algorithms.hpa_jps import HPAJPS
from src.algorithms.hpa_star import HPAStar
from src.algorithms.jps import JPS
from src.core.grid import Grid
from src.core.result import BenchmarkResult
from src.io.map_loader import Scenario, generate_synthetic_map, load_map, load_scenarios


def _benchmark_worker(q, cls_name, grid, alg_kwargs, start, goal):
    try:
        if cls_name == "JPS":
            alg = JPS(grid)
        elif cls_name == "HPAStar":
            alg = HPAStar(grid, cluster_size=alg_kwargs.get("cluster_size", 10))
        elif cls_name == "HPAJPS":
            alg = HPAJPS(grid, cluster_size=alg_kwargs.get("cluster_size", 10), hybrid=alg_kwargs.get("hybrid", True))
        elif cls_name == "AStar":
            alg = AStar(grid)
        else:
            alg = AStar(grid)
        res = alg.search(start, goal)
        q.put(res)
    except Exception as e:  # pragma: no cover - safety in subprocess
        q.put(e)


def run_experiment(
    map_path: str | Path | None,
    algorithms: Sequence[PathfindingAlgorithm],
    scenarios: Sequence[Any],
    grid: Grid | None = None,
    repetitions: int = 30,
) -> list[BenchmarkResult]:
    """Executa os algoritmos para cada mapa, cenário e repetição."""
    results: list[BenchmarkResult] = []
    cached_hierarchy_algorithms: dict[tuple[str, str, str, tuple[int, int]], dict[str, PathfindingAlgorithm]] = {}

    for scenario_config in scenarios:
        scenario_grid = grid
        map_name = "synthetic"
        density_label = "custom"
        if map_path is not None:
            map_name = Path(map_path).stem
        elif isinstance(scenario_config, Scenario):
            map_name = Path(scenario_config.map_name).stem
        if scenario_grid is None:
            if isinstance(scenario_config, dict):
                width = int(scenario_config.get("width", 256))
                height = int(scenario_config.get("height", width))
                density = float(scenario_config.get("density", 0.25))
                scenario_grid = _generate_connected_synthetic_map(width, height, density, seed=42 + int(scenario_config.get("scenario", 0)))
                map_name = scenario_config.get("map_name", map_name)
                density_label = _density_label(density)
            else:
                scenario_grid, map_name, density_label = _resolve_grid(map_path, [scenario_config])
        else:
            if isinstance(scenario_config, dict):
                density = float(scenario_config.get("density", 0.25))
                density_label = _density_label(density)

        start, goal = _coerce_scenario(scenario_config, scenario_grid)
        if isinstance(scenario_config, Scenario):
            scenario_id = scenario_config.bucket
            density_label = "real"
        else:
            scenario_id = int(scenario_config.get("scenario", 0)) if isinstance(scenario_config, dict) else 0
        width = scenario_grid.width
        height = scenario_grid.height
        map_key = (map_name, f"{width}x{height}", density_label, (width, height))

        for algorithm_template in algorithms:
            algorithm_name = algorithm_template.name()
            if algorithm_name in {"HPAStar", "HPAJPS"}:
                cached_algorithms = cached_hierarchy_algorithms.setdefault(map_key, {})
                algorithm = cached_algorithms.get(algorithm_name)
                if algorithm is None:
                    algorithm = _clone_algorithm(algorithm_template, scenario_grid)
                    if hasattr(algorithm, "build_hierarchy"):
                        algorithm.build_hierarchy()  # type: ignore[attr-defined]
                    cached_algorithms[algorithm_name] = algorithm
            else:
                algorithm = _clone_algorithm(algorithm_template, scenario_grid)

            for repetition in range(repetitions):
                # Log progress so long runs are visible to the user.
                print(f"Running {algorithm.name()} on {map_name} scenario={scenario_id} rep={repetition}")

                # Run each search in a separate process and enforce a timeout to
                # avoid hangs/CPU runaway. Timeout seconds can be configured via
                # the BENCH_TIMEOUT_SECS environment variable (default 30s).
                timeout_s = float(os.environ.get("BENCH_TIMEOUT_SECS", 30))

                q: multiprocessing.Queue = multiprocessing.Queue()
                cls_name = algorithm.__class__.__name__
                alg_kwargs = {"cluster_size": getattr(algorithm, "cluster_size", 10), "hybrid": getattr(algorithm, "hybrid", True)}
                proc = multiprocessing.Process(target=_benchmark_worker, args=(q, cls_name, scenario_grid, alg_kwargs, start, goal))
                proc.start()
                proc.join(timeout_s)
                timed_out = False
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    timed_out = True

                if timed_out:
                    print(f"Search timed out: {algorithm.name()} map={map_name} scenario={scenario_id} rep={repetition} (>{timeout_s}s)")
                    # Create a failure BenchmarkResult entry
                    result = None
                    success = False
                    execution_time_ms = float(timeout_s * 1000)
                    preprocessing_time_ms = 0.0
                    nodes_expanded = 0
                    path_length = 0.0
                    path_cost = 0.0
                    num_portals = 0
                    num_jump_points = 0
                    abstract_graph_size = 0
                else:
                    if q.empty():
                        print(f"Search produced no result object for {algorithm.name()} map={map_name} scenario={scenario_id} rep={repetition}")
                        result = None
                        success = False
                        execution_time_ms = 0.0
                        preprocessing_time_ms = 0.0
                        nodes_expanded = 0
                        path_length = 0.0
                        path_cost = 0.0
                        num_portals = 0
                        num_jump_points = 0
                        abstract_graph_size = 0
                    else:
                        res_obj = q.get()
                        if isinstance(res_obj, Exception):
                            # Child raised an exception; surface it and mark failure.
                            print(f"Search subprocess raised: {res_obj!r}")
                            result = None
                            success = False
                            execution_time_ms = 0.0
                            preprocessing_time_ms = 0.0
                            nodes_expanded = 0
                            path_length = 0.0
                            path_cost = 0.0
                            num_portals = 0
                            num_jump_points = 0
                            abstract_graph_size = 0
                        else:
                            result = res_obj
                            success = result.success
                            execution_time_ms = result.execution_time_ms
                            preprocessing_time_ms = result.preprocessing_time_ms
                            nodes_expanded = result.nodes_expanded
                            path_length = result.path_length
                            path_cost = result.path_cost
                            num_portals = getattr(result, "num_portals", 0)
                            num_jump_points = getattr(result, "num_jump_points", 0)
                            abstract_graph_size = getattr(result, "abstract_graph_size", 0)

                results.append(
                    BenchmarkResult(
                        algorithm=algorithm.name(),
                        map_name=map_name,
                        width=width,
                        height=height,
                        density=density_label,
                        scenario=scenario_id * repetitions + repetition,
                        execution_time_ms=execution_time_ms,
                        preprocessing_time_ms=preprocessing_time_ms,
                        nodes_expanded=nodes_expanded,
                        path_length=path_length,
                        path_cost=path_cost,
                        portals=num_portals,
                        jump_points=num_jump_points,
                        abstract_graph_size=abstract_graph_size,
                        success=success,
                    )
                )

    return results


def _resolve_grid(
    map_path: str | Path | None,
    scenarios: Sequence[Any],
) -> tuple[Grid, str, str]:
    if map_path is not None and str(map_path).strip() and Path(map_path).exists():
        grid = load_map(map_path)
        return grid, Path(map_path).stem, "real"

    if scenarios:
        first = scenarios[0]
        if isinstance(first, Scenario):
            width = first.map_width or 256
            height = first.map_height or 256
        elif isinstance(first, dict):
            width = first.get("width", 256)
            height = first.get("height", 256)
        else:
            width = 256
            height = 256
        density = 0.25
        if isinstance(first, dict):
            density = first.get("density", density)
        if isinstance(first, Scenario):
            density = first.optimal_length if density == 0.25 else density
        density_label = _density_label(density)
        grid = _generate_connected_synthetic_map(width, height, density, seed=42)
        return grid, "synthetic", density_label

    grid = _generate_connected_synthetic_map(256, 256, 0.25, seed=42)
    return grid, "synthetic", "medium"


def _coerce_scenario(scenario: Any, grid: Grid) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(scenario, Scenario):
        return scenario.start, scenario.goal
    if isinstance(scenario, dict):
        start = scenario.get("start", (0, 0))
        goal = scenario.get("goal", (grid.width - 1, grid.height - 1))
        return tuple(start), tuple(goal)
    if isinstance(scenario, tuple) and len(scenario) == 2:
        return tuple(scenario[0]), tuple(scenario[1])
    return (0, 0), (grid.width - 1, grid.height - 1)


def _density_label(density: float | str) -> str:
    if isinstance(density, str):
        return density
    if density <= 0.15:
        return "low"
    if density <= 0.35:
        return "medium"
    return "high"


def _generate_connected_synthetic_map(width: int, height: int, density: float, seed: int) -> Grid:
    """Gera um grid sintético garantindo que start e goal permaneçam conectados."""
    for attempt in range(200):
        grid = generate_synthetic_map(width, height, density, seed=seed + attempt)
        if _has_path(grid, (0, 0), (width - 1, height - 1)):
            return grid

    # Fallback determinístico quando a amostragem aleatória não encontra um mapa conectado
    # em poucas tentativas, especialmente para densidades altas.
    fallback = generate_synthetic_map(width, height, density, seed=seed + 10_000)
    if _has_path(fallback, (0, 0), (width - 1, height - 1)):
        return fallback

    walkable = [
        [True if (x == 0 or x == width - 1 or y == 0 or y == height - 1 or (x == width // 2 and y <= height // 2)) else False for x in range(width)]
        for y in range(height)
    ]
    for y in range(height):
        for x in range(width):
            if not walkable[y][x] and (x, y) != (0, 0) and (x, y) != (width - 1, height - 1):
                if (x + y) % 3 != 0:
                    walkable[y][x] = True
    grid = Grid(walkable)
    if _has_path(grid, (0, 0), (width - 1, height - 1)):
        return grid

    raise RuntimeError(f"could not generate a connected synthetic map for size {width}x{height} and density {density}")


def _has_path(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if not grid.is_walkable(*start) or not grid.is_walkable(*goal):
        return False
    queue: deque[tuple[int, int]] = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for nx, ny in grid.neighbors(x, y, allow_diagonal=True):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def build_algorithms(grid: Grid) -> list[PathfindingAlgorithm]:
    """Cria as instâncias dos algoritmos compatíveis com o benchmark."""
    return [
        AStar(grid),
        JPS(grid),
        HPAStar(grid, cluster_size=10),
        HPAJPS(grid, cluster_size=10, hybrid=True),
    ]


def _clone_algorithm(algorithm: PathfindingAlgorithm, grid: Grid) -> PathfindingAlgorithm:
    if isinstance(algorithm, JPS):
        return JPS(grid)
    if isinstance(algorithm, HPAJPS):
        return HPAJPS(grid, cluster_size=getattr(algorithm, "cluster_size", 10), hybrid=getattr(algorithm, "hybrid", True))
    if isinstance(algorithm, HPAStar):
        return HPAStar(grid, cluster_size=getattr(algorithm, "cluster_size", 10))
    if isinstance(algorithm, AStar):
        return AStar(grid)
    return AStar(grid)
