"""Testes adicionais de auditoria para validar componentes centrais do projeto."""

from __future__ import annotations

import math

import pytest

from src.algorithms.astar import AStar
from src.algorithms.hpa_jps import HPAJPS
from src.algorithms.hpa_star import HPAStar
from src.algorithms.jps import JPS
from src.core.grid import Grid
from src.core.heuristics import euclidean, manhattan, octile
from src.core.priority_queue import PriorityQueue
from src.core.result import SearchResult


def test_grid_corner_cutting_blocks_diagonal_when_both_orthogonals_blocked() -> None:
    walkable = [[True] * 3 for _ in range(3)]
    walkable[1][0] = False
    walkable[0][1] = False
    grid = Grid(walkable)

    neighbors = grid.neighbors(0, 0, allow_diagonal=True)

    assert (1, 1) not in neighbors
    assert (0, 1) not in neighbors
    assert (1, 0) not in neighbors


def test_heuristics_values_are_consistent() -> None:
    a = (0, 0)
    b = (3, 4)

    assert octile(a, b) == pytest.approx(3 + 4 + (math.sqrt(2) - 2) * min(3, 4))
    assert euclidean(a, b) == pytest.approx(5.0)
    assert manhattan(a, b) == 7


def test_priority_queue_handles_decrease_key() -> None:
    pq = PriorityQueue[str]()
    pq.push("x", 10.0)
    pq.push("y", 5.0)
    pq.push("x", 3.0)

    assert pq.pop() == "x"
    assert pq.pop() == "y"
    assert pq.is_empty()


def test_search_result_fields_are_present() -> None:
    result = SearchResult(path=[(0, 0), (1, 1)], path_length=1.0, path_cost=math.sqrt(2), nodes_expanded=1, execution_time_ms=0.1, success=True)
    assert result.path == [(0, 0), (1, 1)]
    assert result.num_portals == 0
    assert result.num_jump_points == 0
    assert result.abstract_graph_size == 0
    assert isinstance(result.extra, dict)


def test_hpa_star_and_hpa_jps_preserve_quality_on_open_grid() -> None:
    """HPAStar e HPAJPS devem preservar a qualidade do caminho (custo próximo),
    mas SEM exigir que o grafo abstrato tenha o mesmo tamanho -- pelo contrário,
    o objetivo da integração HPA-JPS é justamente usar uma estrutura de entradas
    diferente (jump points, em vez de só portais geométricos), então espera-se
    que abstract_graph_size e num_jump_points DIVIRJAM entre os dois algoritmos.
    """
    walkable = [[True] * 8 for _ in range(8)]
    grid = Grid(walkable)
    start = (0, 0)
    goal = (7, 7)

    hpa_result = HPAStar(grid, cluster_size=4).search(start, goal)
    hpa_jps_result = HPAJPS(grid, cluster_size=4, hybrid=True).search(start, goal)

    assert hpa_result.success is True
    assert hpa_jps_result.success is True
    assert hpa_jps_result.path_cost == pytest.approx(hpa_result.path_cost, rel=0.1)
    assert hpa_result.num_jump_points == 0
    assert hpa_jps_result.num_jump_points > 0
    assert hpa_result.abstract_graph_size > 0
    assert hpa_jps_result.abstract_graph_size > 0
