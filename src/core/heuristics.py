"""Funções heurísticas puras para estimativa de distância em grids."""

import math


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Distância octile (custo mínimo com movimentos ortogonais e diagonais).

    Adequada para grids com custo 1 (ortogonal) e sqrt(2) (diagonal), pois
    reflete exatamente o custo do caminho ótimo nesse modelo de movimento.

    Args:
        a: Coordenada (x, y) de origem.
        b: Coordenada (x, y) de destino.

    Returns:
        Estimativa admissível do custo mínimo entre ``a`` e ``b``.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # Move o máximo possível na diagonal (custo sqrt(2) cada) e o resto ortogonal.
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Distância euclidiana entre dois pontos do grid.

    Args:
        a: Coordenada (x, y) de origem.
        b: Coordenada (x, y) de destino.

    Returns:
        Distância euclidiana sqrt(dx² + dy²).
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Distância de Manhattan (L1) — apenas movimentos ortogonais.

    Args:
        a: Coordenada (x, y) de origem.
        b: Coordenada (x, y) de destino.

    Returns:
        Soma das diferenças absolutas |dx| + |dy|.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
