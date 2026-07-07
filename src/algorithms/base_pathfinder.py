"""Classe abstrata base para algoritmos de pathfinding."""

from abc import ABC, abstractmethod

from src.core.grid import Grid
from src.core.result import SearchResult


class PathfindingAlgorithm(ABC):
    """Interface comum a todos os algoritmos de busca de caminho."""

    def __init__(self, grid: Grid) -> None:
        """Associa o algoritmo a um grid de navegação.

        Args:
            grid: Instância de ``Grid`` sobre a qual a busca será executada.
        """
        self.grid = grid

    @abstractmethod
    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        """Busca um caminho de ``start`` até ``goal``.

        Args:
            start: Coordenada inicial ``(x, y)``.
            goal: Coordenada objetivo ``(x, y)``.

        Returns:
            ``SearchResult`` com caminho, métricas e indicador de sucesso.
        """
        ...

    def name(self) -> str:
        """Retorna o nome da classe do algoritmo."""
        return self.__class__.__name__
