"""Testes unitários para o algoritmo A*."""

from __future__ import annotations

import math

import pytest

from src.algorithms.astar import AStar
from src.core.grid import Grid


def _empty_grid(size: int = 10) -> Grid:
    """Cria um grid ``size x size`` totalmente transitável."""
    walkable = [[True] * size for _ in range(size)]
    return Grid(walkable)


def _path_cost(grid: Grid, path: list[tuple[int, int]]) -> float:
    """Calcula o custo manual de um caminho célula a célula."""
    total = 0.0
    for i in range(len(path) - 1):
        total += grid.cost(path[i], path[i + 1])
    return total


class TestAStarOpenGrid:
    """Cenários em grid aberto (sem obstáculos)."""

    def test_horizontal_straight_line(self) -> None:
        """10x10 sem obstáculos: caminho reto horizontal deve ser ótimo."""
        grid = _empty_grid(10)
        astar = AStar(grid)
        start = (0, 0)
        goal = (9, 0)

        result = astar.search(start, goal)

        assert result.success is True
        assert result.path[0] == start
        assert result.path[-1] == goal
        # Caminho ótimo: linha reta ao longo de y=0.
        expected_path = [(x, 0) for x in range(10)]
        assert result.path == expected_path
        assert result.path_length == 9.0
        assert result.path_cost == pytest.approx(9.0)
        assert result.nodes_expanded > 0

    def test_diagonal_straight_line(self) -> None:
        """10x10 sem obstáculos: diagonal pura de (0,0) a (9,9) deve ser ótima."""
        grid = _empty_grid(10)
        astar = AStar(grid)
        start = (0, 0)
        goal = (9, 9)

        result = astar.search(start, goal)

        assert result.success is True
        expected_path = [(i, i) for i in range(10)]
        assert result.path == expected_path
        assert result.path_length == 9.0
        assert result.path_cost == pytest.approx(9 * math.sqrt(2))
        assert result.nodes_expanded > 0


class TestAStarWithWall:
    """Cenário com muro vertical forçando desvio pelo único gap."""

    def test_detour_path_cost_matches_manual(self) -> None:
        """Muro em x=5 com gap em (5,5): custo deve bater com caminho manual."""
        size = 10
        walkable = [[True] * size for _ in range(size)]
        # Muro vertical na coluna x=5, exceto célula (5, 5).
        for y in range(size):
            if y != 5:
                walkable[y][5] = False

        grid = Grid(walkable)
        astar = AStar(grid)
        start = (0, 0)
        goal = (9, 0)

        # Custo ótimo manual: (0,0)->(4,4) [4 diag] -> (5,5) [1 diag] ->
        # (9,0) [4 diag + 1 ortogonal] = 9*sqrt(2) + 1.
        expected_cost = 9 * math.sqrt(2) + 1.0

        result = astar.search(start, goal)

        assert result.success is True
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.path_cost == pytest.approx(expected_cost)
        # Confere que o caminho retornado é contíguo e respeita o grid.
        assert _path_cost(grid, result.path) == pytest.approx(result.path_cost)


class TestAStarNoPath:
    """Cenário sem caminho possível entre origem e destino."""

    def test_unreachable_goal(self) -> None:
        """Região do goal isolada: success=False e path vazio."""
        size = 10
        walkable = [[True] * size for _ in range(size)]
        # Isola (9, 9) cercando-o com obstáculos (mantém start acessível em (0,0)).
        for x in range(8, 10):
            for y in range(8, 10):
                if (x, y) != (9, 9):
                    walkable[y][x] = False
        # Barreira adicional impede qualquer rota da origem ao canto inferior direito.
        for x in range(size):
            walkable[7][x] = False

        grid = Grid(walkable)
        astar = AStar(grid)

        result = astar.search((0, 0), (9, 9))

        assert result.success is False
        assert result.path == []
        assert result.path_cost == 0.0
        assert result.path_length == 0.0
