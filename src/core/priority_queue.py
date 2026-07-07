"""Fila de prioridade binária reutilizável com suporte a decrease-key."""

from __future__ import annotations

import heapq
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Wrapper de ``heapq`` com eliminação preguiçosa (lazy deletion).

    Quando um item recebe uma prioridade melhor (decrease-key), basta
    reinseri-lo com a nova prioridade. Entradas obsoletas permanecem no
    heap até serem descartadas no ``pop``, comparando a prioridade armazenada
    com a prioridade vigente do item.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, T]] = []
        # Mapeia cada item à sua melhor prioridade conhecida.
        self._priorities: dict[T, float] = {}
        # Contador monotônico para desempate estável entre entradas iguais.
        self._counter: int = 0

    def push(self, item: T, priority: float) -> None:
        """Insere ``item`` com ``priority`` ou melhora sua prioridade existente.

        Args:
            item: Elemento a armazenar.
            priority: Valor de prioridade (menor = mais urgente).
        """
        # Ignora inserções que não melhoram a prioridade já registrada.
        if item in self._priorities and priority >= self._priorities[item]:
            return

        self._priorities[item] = priority
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> T:
        """Remove e retorna o item de menor prioridade vigente.

        Returns:
            O item com menor prioridade atual.

        Raises:
            IndexError: Se a fila estiver vazia.
        """
        while self._heap:
            priority, _, item = heapq.heappop(self._heap)
            # Descarta entradas obsoletas cuja prioridade no heap não é a atual.
            if self._priorities.get(item) == priority:
                del self._priorities[item]
                return item

        raise IndexError("pop from empty priority queue")

    def is_empty(self) -> bool:
        """Indica se não há itens com prioridade vigente na fila."""
        return len(self._priorities) == 0
