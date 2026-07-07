"""Testes unitários para Jump Point Search (JPS)."""

from __future__ import annotations

import math

import pytest

from src.algorithms.astar import AStar
from src.algorithms.jps import JPS
from src.core.grid import Grid


def _empty_grid(size: int = 10) -> Grid:
    walkable = [[True] * size for _ in range(size)]
    return Grid(walkable)


def _wall_grid(size: int = 10) -> Grid:
    """Grid com muro vertical em x=5 e gap em (5, 5)."""
    walkable = [[True] * size for _ in range(size)]
    for y in range(size):
        if y != 5:
            walkable[y][5] = False
    return Grid(walkable)


def _forced_neighbor_grid() -> Grid:
    """Grid 7x7 com muro vertical em x=3 (y=1..3) gerando forced neighbors."""
    size = 7
    walkable = [[True] * size for _ in range(size)]
    for y in (1, 2, 3):
        walkable[y][3] = False
    return Grid(walkable)


class TestJPSOpenGrid:
    """JPS em grid aberto deve preservar otimalidade de A*."""

    def test_same_cost_as_astar_horizontal(self) -> None:
        grid = _empty_grid(10)
        start = (0, 0)
        goal = (9, 0)

        astar_result = AStar(grid).search(start, goal)
        jps_result = JPS(grid).search(start, goal)

        assert jps_result.success is True
        assert jps_result.path_cost == pytest.approx(astar_result.path_cost)
        assert jps_result.path[0] == start
        assert jps_result.path[-1] == goal

    def test_same_cost_as_astar_diagonal(self) -> None:
        grid = _empty_grid(10)
        start = (0, 0)
        goal = (9, 9)

        astar_result = AStar(grid).search(start, goal)
        jps_result = JPS(grid).search(start, goal)

        assert jps_result.success is True
        assert jps_result.path_cost == pytest.approx(astar_result.path_cost)
        assert jps_result.path_cost == pytest.approx(9 * math.sqrt(2))


class TestJPSForcedNeighbors:
    """Grid com obstáculos que geram forced neighbors."""

    def test_path_cost_matches_astar(self) -> None:
        grid = _wall_grid(10)
        start = (0, 0)
        goal = (9, 0)
        expected_cost = 9 * math.sqrt(2) + 1.0

        astar_result = AStar(grid).search(start, goal)
        jps_result = JPS(grid).search(start, goal)

        assert jps_result.success is True
        assert astar_result.path_cost == pytest.approx(expected_cost)
        assert jps_result.path_cost == pytest.approx(astar_result.path_cost)


class TestJPSNodesExpanded:
    """JPS deve expandir no máximo tantos nós quanto A* no mesmo mapa."""

    def test_jps_expands_fewer_or_equal_nodes(self) -> None:
        grid = _wall_grid(10)
        start = (0, 0)
        goal = (9, 0)

        astar_result = AStar(grid).search(start, goal)
        jps_result = JPS(grid).search(start, goal)

        assert jps_result.success is True
        assert jps_result.nodes_expanded <= astar_result.nodes_expanded

    def test_open_grid_jps_expands_fewer_or_equal(self) -> None:
        grid = _empty_grid(20)
        start = (0, 0)
        goal = (19, 19)

        astar_result = AStar(grid).search(start, goal)
        jps_result = JPS(grid).search(start, goal)

        assert jps_result.path_cost == pytest.approx(astar_result.path_cost)
        assert jps_result.nodes_expanded <= astar_result.nodes_expanded


class TestIdentifyJumpPoints:
    """Teste isolado de ``JPS.identify_jump_points``."""

    def test_vertical_wall_jump_points(self) -> None:
        """Muro em x=3 (y=1..3): JPs esperados ao lado do muro."""
        grid = _forced_neighbor_grid()
        region = (0, 0, 6, 6)

        # Calculados manualmente: células adjacentes ao muro com forced neighbor.
        expected = {
            (2, 1),
            (2, 2),
            (2, 3),
            (4, 1),
            (4, 2),
            (4, 3),
        }

        jump_points = set(JPS.identify_jump_points(grid, region))

        assert expected.issubset(jump_points)
        for point in jump_points:
            x, y = point
            assert 0 <= x <= 6 and 0 <= y <= 6
            assert grid.is_walkable(x, y)
