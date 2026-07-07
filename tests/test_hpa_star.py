"""Testes unitários para HPA*."""

from __future__ import annotations

import pytest

from src.algorithms.astar import AStar
from src.algorithms.hpa_star import HPAStar
from src.core.grid import Grid


def _open_grid() -> Grid:
    walkable = [[True] * 8 for _ in range(8)]
    return Grid(walkable)


class TestHPAStar:
    def test_center_fallback_avoids_unwalkable_cell(self) -> None:
        """Regressão: com 1 único cluster (sem vizinhos) e obstáculo bem no centro
        geométrico, HPAStar falhava (success=False) porque a entrada de fallback
        era o centro geométrico do cluster, sem checar se era transitável. Agora
        deve encontrar uma célula transitável próxima e, com isso, um caminho.
        """
        walkable = [[True] * 10 for _ in range(10)]
        for y in (3, 4, 5):
            walkable[y][4] = False
            walkable[y][5] = False
        grid = Grid(walkable)
        start, goal = (0, 0), (9, 9)

        astar_result = AStar(grid).search(start, goal)
        hpa = HPAStar(grid, cluster_size=10)
        hpa_result = hpa.search(start, goal)

        assert astar_result.success is True
        assert hpa_result.success is True
        assert hpa_result.path_cost == pytest.approx(astar_result.path_cost, rel=0.05)
        for portal_list in hpa._entrances.values():
            for point in portal_list:
                assert grid.is_walkable(*point)

    def test_small_grid_path_cost_is_close_to_astar(self) -> None:
        grid = _open_grid()
        start = (0, 0)
        goal = (7, 7)

        astar_result = AStar(grid).search(start, goal)
        hpa_result = HPAStar(grid, cluster_size=4).search(start, goal)

        assert hpa_result.success is True
        assert hpa_result.path[0] == start
        assert hpa_result.path[-1] == goal
        assert hpa_result.path_cost == pytest.approx(astar_result.path_cost, rel=0.05)

    def test_clusterization_count(self) -> None:
        grid = _open_grid()
        hpa = HPAStar(grid, cluster_size=4)

        clusters = hpa._build_clusters()

        assert len(clusters) == 4

    def test_abstract_graph_size(self) -> None:
        grid = _open_grid()
        result = HPAStar(grid, cluster_size=4).search((0, 0), (7, 7))

        assert result.abstract_graph_size > 0
        assert result.num_portals > 0

    def test_generate_entrances_groups_contiguous_boundary_cells(self) -> None:
        grid = Grid([[True] * 6 for _ in range(6)])
        hpa = HPAStar(grid, cluster_size=3)

        entrances = hpa._generate_entrances()
        bounds = hpa._clusters[0]
        boundary_walkable = [
            (x, y)
            for y in range(bounds[1], bounds[3] + 1)
            for x in range(bounds[0], bounds[2] + 1)
            if grid.is_walkable(x, y) and hpa._is_boundary_cell((x, y), bounds)
        ]

        assert len(entrances[0]) < len(boundary_walkable)

    def test_nodes_expanded_counts_internal_astar_work(self) -> None:
        grid = Grid([[True] * 8 for _ in range(8)])
        result = HPAStar(grid, cluster_size=4).search((0, 0), (7, 7))

        assert result.nodes_expanded > 0
        assert result.nodes_expanded != result.abstract_graph_size

    def test_search_reuses_prebuilt_hierarchy_across_queries(self) -> None:
        grid = Grid([[True] * 8 for _ in range(8)])
        hpa = HPAStar(grid, cluster_size=4)
        build_calls = []
        original_build = hpa._build_abstract_graph

        def wrapped_build() -> dict:
            build_calls.append("build")
            return original_build()

        hpa._build_abstract_graph = wrapped_build  # type: ignore[assignment]

        hpa.search((0, 0), (3, 3))
        hpa.search((4, 4), (7, 7))

        assert len(build_calls) == 1

    def test_search_reports_preprocessing_metrics(self) -> None:
        grid = Grid([[True] * 8 for _ in range(8)])
        result = HPAStar(grid, cluster_size=4).search((0, 0), (7, 7))

        assert result.preprocessing_nodes_expanded >= 0
        assert result.preprocessing_time_ms >= 0.0
