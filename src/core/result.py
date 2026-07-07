"""Resultado padronizado de uma busca de caminho."""

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Métricas e caminho retornados por qualquer algoritmo de pathfinding.

    Attributes:
        path: Sequência de coordenadas (x, y) do início ao objetivo.
        path_length: Número de passos ou distância euclidiana acumulada.
        path_cost: Custo total ponderado (ortogonal=1, diagonal=sqrt(2)).
        nodes_expanded: Quantidade de nós retirados da fila aberta na consulta.
        execution_time_ms: Tempo de execução em milissegundos da consulta.
        preprocessing_time_ms: Tempo de construção da hierarquia em milissegundos.
        preprocessing_nodes_expanded: Nós expandidos durante a construção da hierarquia.
        success: True se um caminho válido foi encontrado.
        num_portals: Preenchido por algoritmos hierárquicos (HPA*).
        num_jump_points: Preenchido por HPA-JPS.
        abstract_graph_size: Tamanho do grafo abstrato (HPA* / HPA-JPS).
        extra: Metadados adicionais específicos do algoritmo.
    """

    path: list[tuple[int, int]]
    path_length: float
    path_cost: float
    nodes_expanded: int
    execution_time_ms: float
    success: bool
    preprocessing_time_ms: float = 0.0
    preprocessing_nodes_expanded: int = 0
    num_portals: int = 0
    num_jump_points: int = 0
    abstract_graph_size: int = 0
    extra: dict = field(default_factory=dict)
