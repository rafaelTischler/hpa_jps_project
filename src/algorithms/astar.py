"""Implementação do algoritmo A* em grid."""

from __future__ import annotations

from collections.abc import Callable

from src.algorithms.base_pathfinder import PathfindingAlgorithm
from src.core.grid import Grid
from src.core.heuristics import octile
from src.core.priority_queue import PriorityQueue
from src.core.result import SearchResult
from src.utils.timing import timed_ms


class AStar(PathfindingAlgorithm):
    """A* clássico com fila de prioridade binária e heurística parametrizável.

    ``get_successors`` é o ponto de extensão para ``JPS``, que substitui
    a expansão de vizinhos pela lógica de jump points.
    """

    def __init__(
        self,
        grid: Grid,
        heuristic: Callable[[tuple[int, int], tuple[int, int]], float] | None = None,
        max_nodes_expanded: int | None = None,
    ) -> None:
        """Inicializa A* com heurística opcional (padrão: octile).

        Args:
            grid: Grid de navegação.
            heuristic: Função ``h(node, goal) -> float``; default ``octile``.
            max_nodes_expanded: Limite de segurança para abortar buscas muito amplas.
        """
        super().__init__(grid)
        self._heuristic = heuristic or octile
        self._max_nodes_expanded = max_nodes_expanded

    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        """Executa A* de ``start`` até ``goal`` e retorna métricas padronizadas."""
        result, elapsed_ms = self._search_impl(start, goal)
        result.execution_time_ms = elapsed_ms
        return result

    @timed_ms
    def _search_impl(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> SearchResult:
        """Núcleo da busca A* (cronometrado por ``timed_ms``)."""
        if not self.grid.is_walkable(*start) or not self.grid.is_walkable(*goal):
            return self._failure_result(nodes_expanded=0)

        if start == goal:
            return self._build_result([start], nodes_expanded=0)

        open_queue: PriorityQueue[tuple[int, int]] = PriorityQueue()
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        open_queue.push(start, self._heuristic(start, goal))
        nodes_expanded = 0

        while not open_queue.is_empty():
            current = open_queue.pop()
            nodes_expanded += 1

            if self._max_nodes_expanded is not None and nodes_expanded > self._max_nodes_expanded:
                return self._failure_result(nodes_expanded)

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return self._build_result(path, nodes_expanded)

            # Subclasses (ex.: JPS) usam _expand_from para poda orientada ao pai.
            self._expand_from = came_from.get(current)
            for successor in self.get_successors(current):
                tentative_g = g_score[current] + self.move_cost(current, successor)

                if successor not in g_score or tentative_g < g_score[successor]:
                    g_score[successor] = tentative_g
                    came_from[successor] = current
                    f_score = tentative_g + self._heuristic(successor, goal)
                    open_queue.push(successor, f_score)

        return self._failure_result(nodes_expanded)

    def move_cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Custo de movimento entre ``a`` e ``b`` (adjacentes por padrão).

        ``JPS`` sobrescreve este método porque seus sucessores são jump points
        possivelmente distantes ao longo de uma linha reta/diagonal.
        """
        return self.grid.cost(a, b)

    def get_successors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        """Retorna vizinhos expansíveis de ``node`` (sobrescrevível por JPS).

        Args:
            node: Coordenada atual ``(x, y)``.

        Returns:
            Lista de células adjacentes transitáveis.
        """
        x, y = node
        return self.grid.neighbors(x, y, allow_diagonal=True)

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstrói o caminho do objetivo até o início usando ``came_from``."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _compute_path_cost(self, path: list[tuple[int, int]]) -> float:
        """Soma os custos de movimento entre células consecutivas do caminho."""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.grid.cost(path[i], path[i + 1])
        return total

    def _build_result(
        self,
        path: list[tuple[int, int]],
        nodes_expanded: int,
    ) -> SearchResult:
        """Monta ``SearchResult`` de sucesso a partir do caminho encontrado."""
        path_cost = self._compute_path_cost(path)
        return SearchResult(
            path=path,
            path_length=float(len(path) - 1),
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            execution_time_ms=0.0,
            success=True,
        )

    def _failure_result(self, nodes_expanded: int) -> SearchResult:
        """Monta ``SearchResult`` indicando que nenhum caminho foi encontrado."""
        return SearchResult(
            path=[],
            path_length=0.0,
            path_cost=0.0,
            nodes_expanded=nodes_expanded,
            execution_time_ms=0.0,
            success=False,
        )
