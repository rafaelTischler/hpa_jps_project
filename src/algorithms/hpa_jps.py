"""HPA-JPS: substitui portais geométricos por jump points identificados por JPS."""

from __future__ import annotations

from typing import Any

from src.algorithms.hpa_star import HPAStar
from src.algorithms.jps import JPS
from src.core.grid import Grid
from src.core.result import SearchResult


class HPAJPS(HPAStar):
    """HPA* com entradas derivadas de jump points estruturais do JPS."""

    def __init__(self, grid: Grid, cluster_size: int = 10, hybrid: bool = True) -> None:
        self.hybrid = hybrid
        self._jump_points_used = 0
        self._jump_point_cache: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
        super().__init__(grid, cluster_size=cluster_size)

    def build_hierarchy(self) -> dict[tuple[Any, ...], dict[tuple[Any, ...], float]]:
        """Reconstrói a hierarquia apenas quando o grid muda e invalida o cache de jump points."""
        signature = self._grid_signature()
        if self._hierarchy_ready and self._hierarchy_signature == signature:
            return self._abstract_graph
        self._jump_point_cache.clear()
        return super().build_hierarchy()

    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        """Executa HPA-JPS preservando o número real de jump points usados como entradas."""
        result = super().search(start, goal)
        result.num_jump_points = self._jump_points_used
        return result

    def _generate_entrances(self) -> dict[int, list[tuple[int, int]]]:
        """Gera entradas combinando jump points com portais tradicionais quando hybrid=True."""
        if not self._clusters:
            self._build_clusters()

        self._jump_points_used = 0
        entrances: dict[int, list[tuple[int, int]]] = {}
        for cluster_id, bounds in enumerate(self._clusters):
            jump_points = self._jump_point_cache.get(bounds)
            if jump_points is None:
                jump_points = [
                    point
                    for point in JPS.identify_jump_points(self.grid, bounds)
                    if self._is_boundary_cell(point, bounds)
                ]
                self._jump_point_cache[bounds] = jump_points
            traditional_portals = self._collect_boundary_segment_representatives(cluster_id, bounds)
            if self.hybrid:
                combined = self._dedupe_points(jump_points + traditional_portals)
            else:
                # hybrid=False usa exclusivamente jump points; alguns clusters podem ficar sem entradas.
                combined = self._dedupe_points(jump_points)
            entrances[cluster_id] = combined
            jump_point_set = set(jump_points)
            self._jump_points_used += len([point for point in combined if point in jump_point_set])

        self._entrances = entrances
        return entrances
