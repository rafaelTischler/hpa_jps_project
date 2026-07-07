"""Execução de experimentos de benchmark para os algoritmos de pathfinding."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence

from src.algorithms.astar import AStar
from src.algorithms.base_pathfinder import PathfindingAlgorithm
from src.algorithms.hpa_jps import HPAJPS
from src.algorithms.hpa_star import HPAStar
from src.algorithms.jps import JPS
from src.core.grid import Grid
from src.io.map_loader import Scenario, generate_synthetic_map, load_map


def run_experiment(
    map_path: str | Path | None,
    algorithms: Sequence[PathfindingAlgorithm],
    scenarios: Sequence[Any],
    grid: Grid | None = None,
) -> list[dict[str, Any]]:
    """Executa todos os algoritmos sobre os mesmos pares start/goal do mesmo mapa.

    O ``map_path`` pode apontar para um arquivo ``.map`` real ou para um mapa sintético
    gerado quando o arquivo não existir. Cada cenário pode ser um ``Scenario`` da
    camada de I/O, um ``dict`` com ``start``/``goal`` ou uma tupla ``(start, goal)``.
    """
    results: list[dict[str, Any]] = []
    cached_hierarchy_algorithms: dict[tuple[str, str, str, tuple[int, int]], dict[str, PathfindingAlgorithm]] = {}

    for run_id, scenario in enumerate(scenarios):
        scenario_grid = grid
        map_name = "synthetic"
        density_label = "custom"
        if scenario_grid is None:
            if isinstance(scenario, dict):
                width = int(scenario.get("width", 256))
                height = int(scenario.get("height", width))
                density = float(scenario.get("density", 0.25))
                scenario_grid = _generate_connected_synthetic_map(width, height, density, seed=42 + run_id)
                map_name = "synthetic"
                density_label = _density_label(density)
            else:
                scenario_grid, map_name, density_label = _resolve_grid(map_path, [scenario])
        else:
            if isinstance(scenario, dict):
                density = float(scenario.get("density", 0.25))
                density_label = _density_label(density)
        start, goal = _coerce_scenario(scenario, scenario_grid)
        map_key = (map_name, f"{scenario_grid.width}x{scenario_grid.height}", density_label, (scenario_grid.width, scenario_grid.height))
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
            result = algorithm.search(start, goal)
            results.append(
                {
                    "map_name": map_name,
                    "size": f"{scenario_grid.width}x{scenario_grid.height}",
                    "density": density_label,
                    "algorithm": algorithm.name(),
                    "run_id": run_id,
                    "execution_time_ms": result.execution_time_ms,
                    "nodes_expanded": result.nodes_expanded,
                    "path_length": result.path_length,
                    "path_cost": result.path_cost,
                    "num_portals": result.num_portals,
                    "num_jump_points": result.num_jump_points,
                    "abstract_graph_size": result.abstract_graph_size,
                    "preprocessing_time_ms": result.preprocessing_time_ms,
                    "preprocessing_nodes_expanded": result.preprocessing_nodes_expanded,
                    "success": result.success,
                }
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
