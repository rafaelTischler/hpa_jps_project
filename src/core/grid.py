"""Representação de grid 2D para pathfinding."""

from __future__ import annotations

import math
from typing import Sequence

# Deslocamentos ortogonais e diagonais (dx, dy).
_ORTHOGONAL = ((0, 1), (1, 0), (0, -1), (-1, 0))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class Grid:
    """Grid 2D com células transitáveis ou bloqueadas por obstáculos.

    A matriz interna usa convenção ``walkable[y][x]``: True = transitável,
    False = obstáculo.
    """

    def __init__(self, walkable: Sequence[Sequence[bool]]) -> None:
        """Inicializa o grid a partir de uma matriz booleana.

        Args:
            walkable: Matriz ``height x width``; True indica célula transitável.

        Raises:
            ValueError: Se a matriz estiver vazia ou com linhas de tamanhos distintos.
        """
        if not walkable or not walkable[0]:
            raise ValueError("walkable matrix must be non-empty")

        width = len(walkable[0])
        for row in walkable:
            if len(row) != width:
                raise ValueError("all rows in walkable matrix must have the same width")

        self._walkable = [list(row) for row in walkable]
        self.height = len(self._walkable)
        self.width = width

    def is_walkable(self, x: int, y: int) -> bool:
        """Verifica se a célula ``(x, y)`` está dentro dos limites e é transitável."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self._walkable[y][x]

    def neighbors(
        self,
        x: int,
        y: int,
        allow_diagonal: bool = True,
    ) -> list[tuple[int, int]]:
        """Retorna vizinhos transitáveis de ``(x, y)``.

        Quando ``allow_diagonal`` é True, aplica regra de *corner cutting*:
        movimento diagonal é bloqueado se **ambas** as células ortogonais
        adjacentes ao canto forem obstáculo (impede "cortar" o vértice de
        um canto bloqueado).

        Args:
            x: Coluna da célula.
            y: Linha da célula.
            allow_diagonal: Se True, inclui vizinhos diagonais válidos.

        Returns:
            Lista de coordenadas ``(nx, ny)`` alcançáveis a partir de ``(x, y)``.
        """
        result: list[tuple[int, int]] = []

        for dx, dy in _ORTHOGONAL:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                result.append((nx, ny))

        if not allow_diagonal:
            return result

        for dx, dy in _DIAGONAL:
            nx, ny = x + dx, y + dy
            if not self.is_walkable(nx, ny):
                continue

            # Células ortogonais que compartilham o canto com o destino diagonal.
            ortho_a = (x + dx, y)
            ortho_b = (x, y + dy)
            # Bloqueia diagonal apenas se os dois ortogonais forem obstáculo.
            if not self.is_walkable(ortho_a[0], ortho_a[1]) and not self.is_walkable(
                ortho_b[0], ortho_b[1]
            ):
                continue

            result.append((nx, ny))

        return result

    def cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Custo de movimento entre duas células adjacentes.

        Args:
            a: Célula de origem ``(x, y)``.
            b: Célula de destino ``(x, y)`` (vizinha de ``a``).

        Returns:
            1.0 para passo ortogonal, ``sqrt(2)`` para passo diagonal.

        Raises:
            ValueError: Se ``b`` não for vizinha válida de ``a``.
        """
        ax, ay = a
        bx, by = b
        dx = abs(ax - bx)
        dy = abs(ay - by)

        if dx + dy == 1:
            return 1.0
        if dx == 1 and dy == 1:
            return math.sqrt(2)

        raise ValueError(f"cells {a} and {b} are not adjacent neighbors")
