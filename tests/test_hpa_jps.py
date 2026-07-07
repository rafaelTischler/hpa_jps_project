"""Testes unitários para HPA-JPS."""

from __future__ import annotations

import pytest

from src.algorithms.hpa_jps import HPAJPS
from src.algorithms.hpa_star import HPAStar
from src.core.grid import Grid


def _open_grid() -> Grid:
    walkable = [[True] * 8 for _ in range(8)]
    return Grid(walkable)


class TestHPAJPS:
    def test_hpa_jps_path_cost_is_close_to_hpa_star(self) -> None:
        grid = _open_grid()
        start = (0, 0)
        goal = (7, 7)

        hpa_result = HPAStar(grid, cluster_size=4).search(start, goal)
        hpa_jps_result = HPAJPS(grid, cluster_size=4).search(start, goal)

        assert hpa_jps_result.success is True
        assert hpa_jps_result.path_cost == pytest.approx(hpa_result.path_cost, rel=0.1)
        assert hpa_jps_result.num_jump_points >= 0

    def test_hybrid_false_still_supports_search(self) -> None:
        grid = _open_grid()
        result = HPAJPS(grid, cluster_size=4, hybrid=False).search((0, 0), (7, 7))

        assert result.success is True
        assert result.path_cost > 0.0

    def test_hybrid_true_uses_jump_points_as_entries(self) -> None:
        grid = _open_grid()
        result = HPAJPS(grid, cluster_size=4, hybrid=True).search((0, 0), (7, 7))

        assert result.success is True
        assert result.num_jump_points > 0

    def test_search_reuses_hierarchy_across_queries(self) -> None:
        grid = _open_grid()
        hpa_jps = HPAJPS(grid, cluster_size=4)
        build_calls = []
        original_build = hpa_jps._build_abstract_graph

        def wrapped_build() -> dict:
            build_calls.append("build")
            return original_build()

        hpa_jps._build_abstract_graph = wrapped_build  # type: ignore[assignment]

        hpa_jps.search((0, 0), (3, 3))
        hpa_jps.search((4, 4), (7, 7))

        assert len(build_calls) == 1
