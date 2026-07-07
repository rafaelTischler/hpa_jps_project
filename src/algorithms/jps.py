"""Jump Point Search (JPS) — extensão de A* com poda de simetria.

Implementação baseada em Harabor & Grastien (2012, 2014): natural neighbors,
forced neighbors e busca recursiva de jump points substituem a expansão
célula-a-célula de A*, preservando otimalidade no grid octile.
"""

from __future__ import annotations

import math

from src.algorithms.astar import AStar
from src.core.grid import Grid
from src.core.heuristics import octile
from src.core.result import SearchResult

# Oito direções de movimento (ortogonais + diagonais).
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
    (1, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
)


class JPS(AStar):
    """Jump Point Search reaproveitando fila, heurística e reconstrução de A*."""

    def __init__(self, grid: Grid, heuristic=None) -> None:
        super().__init__(grid, heuristic=heuristic)
        self._start: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (0, 0)

    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        """Executa JPS registrando origem/destino para a busca."""
        self._start = start
        self._goal = goal
        return super().search(start, goal)

    def move_cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Custo octile ao longo do salto reto entre jump points distantes."""
        return octile(a, b)

    def _build_result(
        self,
        path: list[tuple[int, int]],
        nodes_expanded: int,
    ) -> SearchResult:
        """Expande jump points em um caminho celular válido para o cálculo de custo."""
        expanded_path = self._expand_path(path)
        path_cost = self._compute_path_cost(expanded_path)
        return SearchResult(
            path=expanded_path,
            path_length=float(len(expanded_path) - 1),
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            execution_time_ms=0.0,
            success=True,
        )

    def _expand_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Converte uma sequência de jump points em um caminho real de células adjacentes."""
        if len(path) <= 1:
            return path

        expanded: list[tuple[int, int]] = [path[0]]
        for previous, current in zip(path, path[1:]):
            if previous == current:
                continue
            try:
                if self.grid.cost(previous, current) in {1.0, math.sqrt(2)}:
                    expanded.append(current)
                    continue
            except ValueError:
                pass
            dx = 0 if previous[0] == current[0] else (1 if current[0] > previous[0] else -1)
            dy = 0 if previous[1] == current[1] else (1 if current[1] > previous[1] else -1)
            x, y = previous
            while (x, y) != current:
                nx, ny = x + dx, y + dy
                if not self.grid.is_walkable(nx, ny):
                    break
                x, y = nx, ny
                if (x, y) != current:
                    expanded.append((x, y))
            expanded.append(current)

        return expanded

    def get_successors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        """Retorna sucessores válidos seguindo a estratégia de jump points."""
        parent = getattr(self, "_expand_from", None)
        directions = self._pruned_directions(parent, node)
        successors: list[tuple[int, int]] = []
        for direction in directions:
            successor = self._jump(node, direction, self._start, self._goal)
            if successor is not None:
                successors.append(successor)
        return self._unique(successors)

    def _unique(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        unique: list[tuple[int, int]] = []
        for point in points:
            if point not in unique:
                unique.append(point)
        return unique

    def _pruned_directions(
        self,
        parent: tuple[int, int] | None,
        node: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Calcula direções de busca após poda de simetria.

        Bug corrigido: além da continuação natural (3 direções derivadas do
        movimento do pai), o caso diagonal precisa também considerar direções
        FORÇADAS por obstáculos locais em ``node`` -- sem isso, ao chegar num
        ponto de passagem estreito (ex.: gap num muro), a busca nunca explora a
        direção necessária para "virar" e contornar o obstáculo, mesmo que esse
        ponto já tenha sido corretamente identificado como jump point.
        """
        if parent is None:
            return list(_DIRECTIONS)
        x, y = node
        dx = _sign(node[0] - parent[0])
        dy = _sign(node[1] - parent[1])
        if dx != 0 and dy != 0:
            directions = [(dx, dy), (dx, 0), (0, dy)]
            if not self.grid.is_walkable(x + dx, y) and self.grid.is_walkable(x, y + dy):
                directions.append((-dx, dy))
            if not self.grid.is_walkable(x, y + dy) and self.grid.is_walkable(x + dx, y):
                directions.append((dx, -dy))
            return self._unique(directions)
        if dx != 0:
            return [(dx, 0), (dx, 1), (dx, -1)]
        return [(0, dy), (1, dy), (-1, dy)]

    def _jump(
        self,
        node: tuple[int, int],
        direction: tuple[int, int],
        start: tuple[int, int],
        goal: tuple[int, int],
        depth: int = 0,
    ) -> tuple[int, int] | None:
        """Avança em uma direção até encontrar um jump point válido."""
        dx, dy = direction
        x, y = node
        nx, ny = x + dx, y + dy
        if not self.grid.is_walkable(nx, ny):
            return None
        # Safety guard: avoid runaway recursion by bounding depth to a
        # reasonable upper limit based on grid size. This prevents hangs if
        # the jump logic enters a pathological loop.
        max_depth = self.grid.width * self.grid.height
        if depth > max_depth:
            return None

        current = (nx, ny)
        if current == goal:
            return current
        if self._is_forced_neighbor(current, direction):
            return current

        if dx != 0 and dy != 0:
            if self._jump(current, (dx, 0), start, goal, depth + 1) is not None:
                return current
            if self._jump(current, (0, dy), start, goal, depth + 1) is not None:
                return current

        return self._jump(current, direction, start, goal, depth + 1)

    def _has_complex_obstacles(self) -> bool:
        """Indica se o grid possui obstáculos que exigem fallback para A* para preservar corretude."""
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if not self.grid.is_walkable(x, y):
                    continue
                if self._is_forced_neighbor((x, y), (1, 0)) or self._is_forced_neighbor((x, y), (0, 1)):
                    return True
        return False

    def _is_forced_neighbor(self, pos: tuple[int, int], direction: tuple[int, int]) -> bool:
        """Detecta vizinho forçado na posição atual, conforme a poda de JPS."""
        x, y = pos
        dx, dy = direction
        if dx != 0 and dy == 0:
            return (
                self.grid.is_walkable(x, y + 1)
                and not self.grid.is_walkable(x + dx, y + 1)
            ) or (
                self.grid.is_walkable(x, y - 1)
                and not self.grid.is_walkable(x + dx, y - 1)
            )
        if dx == 0 and dy != 0:
            return (
                self.grid.is_walkable(x + 1, y)
                and not self.grid.is_walkable(x + 1, y + dy)
            ) or (
                self.grid.is_walkable(x - 1, y)
                and not self.grid.is_walkable(x - 1, y + dy)
            )
        if dx != 0 and dy != 0:
            return (
                not self.grid.is_walkable(x + dx, y)
                and self.grid.is_walkable(x, y + dy)
            ) or (
                not self.grid.is_walkable(x, y + dy)
                and self.grid.is_walkable(x + dx, y)
            )
        return False

    @staticmethod
    def identify_jump_points(
        grid: Grid,
        region: tuple[int, int, int, int],
    ) -> list[tuple[int, int]]:
        """Identifica jump points estruturais numa sub-região, sem buscar caminho."""
        x_min, y_min, x_max, y_max = region
        jump_points: list[tuple[int, int]] = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if not grid.is_walkable(x, y):
                    continue
                if JPS._is_structural_jump_point(grid, x, y):
                    jump_points.append((x, y))
        return jump_points

    @staticmethod
    def _is_structural_jump_point(grid: Grid, x: int, y: int) -> bool:
        """Marca uma célula como jump point estrutural se tiver vizinho forçado."""
        for direction in _DIRECTIONS:
            if JPS._has_forced_static(grid, (x, y), direction):
                return True
        return False

    @staticmethod
    def _has_forced_static(
        grid: Grid,
        pos: tuple[int, int],
        direction: tuple[int, int],
    ) -> bool:
        """Versão estática de ``_is_forced_neighbor`` para ``identify_jump_points``."""
        x, y = pos
        dx, dy = direction
        if dx != 0 and dy == 0:
            return (
                not grid.is_walkable(x + dx, y + 1)
                and grid.is_walkable(x, y + 1)
            ) or (
                not grid.is_walkable(x + dx, y - 1)
                and grid.is_walkable(x, y - 1)
            )
        if dx == 0 and dy != 0:
            return (
                not grid.is_walkable(x + 1, y + dy)
                and grid.is_walkable(x + 1, y)
            ) or (
                not grid.is_walkable(x - 1, y + dy)
                and grid.is_walkable(x - 1, y)
            )
        if dx != 0 and dy != 0:
            return (
                not grid.is_walkable(x + dx, y)
                and grid.is_walkable(x, y + dy)
            ) or (
                not grid.is_walkable(x, y + dy)
                and grid.is_walkable(x + dx, y)
            )
        return False


def _sign(value: int) -> int:
    """Retorna -1, 0 ou 1 conforme o sinal de ``value``."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
