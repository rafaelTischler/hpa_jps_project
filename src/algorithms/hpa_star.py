"""Implementação simplificada de HPA* usando A* interno para buscas locais."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any

from src.algorithms.astar import AStar
from src.algorithms.base_pathfinder import PathfindingAlgorithm
from src.core.grid import Grid
from src.core.result import SearchResult
from src.core.priority_queue import PriorityQueue


class BoundedGridView:
    """View do grid original limitada a uma região retangular.

    O wrapper delega toda a lógica ao grid original, mas marca células fora dos
    limites como não transitáveis. Isso mantém a mesma assinatura de ``Grid``
    sem copiar a matriz inteira.
    """

    def __init__(self, base_grid: Grid, x0: int, y0: int, x1: int, y1: int) -> None:
        self._base_grid = base_grid
        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1
        self.width = base_grid.width
        self.height = base_grid.height

    def is_walkable(self, x: int, y: int) -> bool:
        if not (self._x0 <= x <= self._x1 and self._y0 <= y <= self._y1):
            return False
        return self._base_grid.is_walkable(x, y)

    def neighbors(self, x: int, y: int, allow_diagonal: bool = True) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                result.append((nx, ny))

        if not allow_diagonal:
            return result

        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not self.is_walkable(nx, ny):
                continue

            ortho_a = (x + dx, y)
            ortho_b = (x, y + dy)
            if not self.is_walkable(ortho_a[0], ortho_a[1]) and not self.is_walkable(ortho_b[0], ortho_b[1]):
                continue

            result.append((nx, ny))

        return result

    def cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return self._base_grid.cost(a, b)


class HPAStar(PathfindingAlgorithm):
    """Hierarchical Pathfinding A* com clusters fixos e portais geométricos."""

    def __init__(self, grid: Grid, cluster_size: int = 10) -> None:
        super().__init__(grid)
        if cluster_size <= 0:
            raise ValueError("cluster_size must be positive")
        self.cluster_size = cluster_size
        self._clusters: list[tuple[int, int, int, int]] = []
        self._cluster_lookup: dict[tuple[int, int], int] = {}
        self._entrances: dict[int, list[tuple[int, int]]] = {}
        self._abstract_graph: dict[tuple[Any, ...], dict[tuple[Any, ...], float]] = {}
        self._local_path_cost_cache: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}
        self._hierarchy_ready = False
        self._hierarchy_signature: tuple[tuple[bool, ...], ...] | None = None
        self._preprocessing_time_ms = 0.0
        self._preprocessing_nodes_expanded = 0
        self._query_nodes_expanded = 0
        self._active_metric_scope = "query"
        self._last_nodes_expanded = 0
        self._local_astar_limit = self.grid.width * self.grid.height * 4
        self._local_astar = AStar(grid, max_nodes_expanded=self._local_astar_limit)

    def _build_clusters(self) -> list[tuple[int, int, int, int]]:
        """Particiona o grid em clusters regulares de tamanho ``cluster_size``."""
        self._clusters = []
        self._cluster_lookup = {}
        for y0 in range(0, self.grid.height, self.cluster_size):
            for x0 in range(0, self.grid.width, self.cluster_size):
                x_max = min(self.grid.width - 1, x0 + self.cluster_size - 1)
                y_max = min(self.grid.height - 1, y0 + self.cluster_size - 1)
                bounds = (x0, y0, x_max, y_max)
                self._clusters.append(bounds)
                for y in range(y0, y_max + 1):
                    for x in range(x0, x_max + 1):
                        self._cluster_lookup[(x, y)] = len(self._clusters) - 1
        return self._clusters

    def _generate_entrances(self) -> dict[int, list[tuple[int, int]]]:
        """Gera entradas por segmento de fronteira, em vez de uma entrada por célula.

        Bug corrigido: os fallbacks (canto do cluster e centro geométrico) eram
        adicionados sem checar se a célula era transitável. Em qualquer mapa com
        obstáculo bem no meio (ou no canto) de um cluster -- bastante comum em
        mapas reais -- isso criava uma entrada "morta", que nunca conecta a nada,
        e podia deixar um cluster totalmente sem entrada válida (fazendo o HPA*
        falhar mesmo quando existe caminho, como confirmado em teste).
        """
        if not self._clusters:
            self._build_clusters()

        entrances: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(self._clusters))}
        for cluster_id, bounds in enumerate(self._clusters):
            candidates = self._collect_boundary_segment_representatives(cluster_id, bounds)
            candidates = [p for p in candidates if self.grid.is_walkable(*p)]

            center = self._find_walkable_cell_in_bounds(bounds)
            if center is not None and center not in candidates:
                candidates.append(center)

            if not candidates:
                # Cluster sem nenhuma célula transitável (ex.: todo bloqueado por
                # obstáculos) -- não há entrada possível, e não há problema nisso:
                # nenhum caminho pode passar por um cluster totalmente sólido.
                candidates = []

            entrances[cluster_id] = self._dedupe_points(candidates)
        self._entrances = entrances
        return entrances

    def _find_walkable_cell_in_bounds(self, bounds: tuple[int, int, int, int]) -> tuple[int, int] | None:
        """Acha uma célula transitável dentro do cluster, preferindo o centro geométrico.

        Tenta o centro exato primeiro (caso comum); se não for transitável, faz uma
        varredura em espiral crescente a partir do centro até achar alguma célula
        livre, ou retorna ``None`` se o cluster inteiro for obstáculo.
        """
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if self.grid.is_walkable(cx, cy):
            return (cx, cy)

        max_radius = max(x1 - x0, y1 - y0) + 1
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    x, y = cx + dx, cy + dy
                    if x0 <= x <= x1 and y0 <= y <= y1 and self.grid.is_walkable(x, y):
                        return (x, y)
        return None

    def _grid_signature(self) -> tuple[tuple[bool, ...], ...]:
        """Retorna uma assinatura estrutural do grid para evitar reconstruções desnecessárias."""
        return tuple(tuple(row) for row in self.grid._walkable)

    def build_hierarchy(self) -> dict[tuple[Any, ...], dict[tuple[Any, ...], float]]:
        """Constrói e cacheia a hierarquia abstrata do mapa uma única vez por grid."""
        signature = self._grid_signature()
        if self._hierarchy_ready and self._hierarchy_signature == signature:
            return self._abstract_graph

        started_at = time.perf_counter()
        self._active_metric_scope = "preprocessing"
        self._preprocessing_nodes_expanded = 0
        self._preprocessing_time_ms = 0.0
        self._local_path_cost_cache.clear()

        self._build_clusters()
        self._generate_entrances()
        self._build_abstract_graph()

        self._hierarchy_ready = True
        self._hierarchy_signature = signature
        self._preprocessing_time_ms = (time.perf_counter() - started_at) * 1000.0
        self._active_metric_scope = "query"
        self._query_nodes_expanded = 0
        self._last_nodes_expanded = 0
        return self._abstract_graph

    def _bounded_region_for_cluster(self, cluster_id: int, margin: int = 2) -> tuple[int, int, int, int]:
        """Região limitada em torno de um único cluster, com margem de segurança."""
        x0, y0, x1, y1 = self._clusters[cluster_id]
        return (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(self.grid.width - 1, x1 + margin),
            min(self.grid.height - 1, y1 + margin),
        )

    def _multi_target_distances(
        self,
        origin: tuple[int, int],
        region: tuple[int, int, int, int],
        targets: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """Roda UMA busca (Dijkstra) a partir de ``origin`` e retorna o custo até
        cada ponto em ``targets`` alcançado, dentro da região limitada.

        Isto substitui rodar uma busca A* separada para CADA par (origem, alvo):
        em vez de O(n²) buscas por cluster (uma por par de entradas), fazemos
        O(n) buscas (uma por entrada de origem), cada uma computando a distância
        para TODAS as outras entradas de uma só vez. O custo de cada busca
        individual permanece limitado à região (bounded), então o ganho é apenas
        na quantidade de buscas, não no escopo de cada uma.
        """
        remaining = {t for t in targets if t != origin}
        if not remaining:
            return {}

        bounded_grid = BoundedGridView(self.grid, *region)
        if not bounded_grid.is_walkable(*origin):
            return {}

        pq: PriorityQueue[tuple[int, int]] = PriorityQueue()
        pq.push(origin, 0.0)
        best_cost: dict[tuple[int, int], float] = {origin: 0.0}
        found: dict[tuple[int, int], float] = {}
        visited: set[tuple[int, int]] = set()

        while not pq.is_empty() and remaining:
            current = pq.pop()
            if current in visited:
                continue
            visited.add(current)
            self._record_nodes_expanded()
            current_cost = best_cost[current]

            if current in remaining:
                found[current] = current_cost
                remaining.discard(current)
                if not remaining:
                    break

            for neighbor in bounded_grid.neighbors(*current, allow_diagonal=True):
                if neighbor in visited:
                    continue
                candidate = current_cost + bounded_grid.cost(current, neighbor)
                if candidate < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = candidate
                    pq.push(neighbor, candidate)

        return found

    def _build_abstract_graph(self) -> dict[tuple[Any, ...], dict[tuple[Any, ...], float]]:
        """Cria um grafo abstrato entre portais e entradas de cluster.

        Usa ``_multi_target_distances`` (uma busca por origem, cobrindo vários
        alvos de uma vez) em vez de uma busca A* separada por PAR de entradas --
        essa é a otimização que evita a explosão O(n²) de buscas locais.
        """
        if not self._clusters:
            self._build_clusters()
        entrances = self._entrances if self._entrances else self._generate_entrances()
        graph: dict[tuple[Any, ...], dict[tuple[Any, ...], float]] = defaultdict(dict)

        for cluster_id, portal_nodes in entrances.items():
            for portal in portal_nodes:
                node_id = ("portal", cluster_id, portal[0], portal[1])
                graph[node_id] = {}

        # Uma única busca por entrada de ORIGEM, cobrindo de uma vez: as demais
        # entradas do MESMO cluster + as entradas de TODOS os clusters vizinhos.
        # Antes eram até (1 intra + até 4 vizinhos) = até 5 buscas por origem;
        # agora é sempre 1 busca por origem, sobre uma região um pouco maior
        # (cluster + vizinhos). O guard "neighbor_id > cluster_id" ao montar os
        # alvos de vizinhos evita computar a mesma aresta duas vezes (uma a
        # partir de cada lado do par de clusters).
        for cluster_id, portal_nodes in entrances.items():
            if not portal_nodes:
                continue
            neighbor_ids = [nid for nid in self._adjacent_cluster_ids(cluster_id) if nid > cluster_id]

            bounds = self._clusters[cluster_id]
            x0, y0, x1, y1 = bounds
            for nid in neighbor_ids:
                nb = self._clusters[nid]
                x0 = min(x0, nb[0])
                y0 = min(y0, nb[1])
                x1 = max(x1, nb[2])
                y1 = max(y1, nb[3])
            region = (
                max(0, x0 - 2),
                max(0, y0 - 2),
                min(self.grid.width - 1, x1 + 2),
                min(self.grid.height - 1, y1 + 2),
            )

            for index, origin in enumerate(portal_nodes):
                targets = list(portal_nodes[index + 1 :])
                for nid in neighbor_ids:
                    targets.extend(entrances.get(nid, []))
                if not targets:
                    continue
                distances = self._multi_target_distances(origin, region, targets)
                node_a = ("portal", cluster_id, origin[0], origin[1])
                for target, cost in distances.items():
                    target_cluster = self._cluster_lookup.get(target, cluster_id)
                    node_b = ("portal", target_cluster, target[0], target[1])
                    graph[node_a][node_b] = cost
                    graph[node_b][node_a] = cost

        self._abstract_graph = dict(graph)
        return self._abstract_graph

    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        """Executa HPA* compondo buscas locais com A* no grafo abstrato."""
        self.build_hierarchy()
        started_at = time.perf_counter()
        self._active_metric_scope = "query"
        self._query_nodes_expanded = 0
        self._last_nodes_expanded = 0
        graph = self._abstract_graph
        entrances = self._entrances
        start_cluster = self._cluster_id_for_point(start)
        goal_cluster = self._cluster_id_for_point(goal)

        start_node = ("start", start[0], start[1])
        goal_node = ("goal", goal[0], goal[1])

        adjusted_graph = {node: dict(neigh) for node, neigh in graph.items()}
        adjusted_graph[start_node] = {}
        adjusted_graph[goal_node] = {}

        start_entrances = entrances.get(start_cluster, [])
        if start_entrances:
            start_region = self._bounded_region_for_cluster(start_cluster)
            start_distances = self._multi_target_distances(start, start_region, start_entrances)
            for portal, cost in start_distances.items():
                portal_node = ("portal", start_cluster, portal[0], portal[1])
                adjusted_graph.setdefault(portal_node, {})
                adjusted_graph[start_node][portal_node] = cost
                adjusted_graph[portal_node][start_node] = cost

        goal_entrances = entrances.get(goal_cluster, [])
        if goal_entrances:
            goal_region = self._bounded_region_for_cluster(goal_cluster)
            goal_distances = self._multi_target_distances(goal, goal_region, goal_entrances)
            for portal, cost in goal_distances.items():
                portal_node = ("portal", goal_cluster, portal[0], portal[1])
                adjusted_graph.setdefault(portal_node, {})
                adjusted_graph[goal_node][portal_node] = cost
                adjusted_graph[portal_node][goal_node] = cost

        abstract_path = self._shortest_path(adjusted_graph, start_node, goal_node)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        nodes_expanded_total = self._query_nodes_expanded
        if not abstract_path:
            return SearchResult(
                path=[],
                path_length=0.0,
                path_cost=0.0,
                nodes_expanded=nodes_expanded_total,
                execution_time_ms=elapsed_ms,
                preprocessing_time_ms=self._preprocessing_time_ms,
                preprocessing_nodes_expanded=self._preprocessing_nodes_expanded,
                success=False,
                num_portals=len(self._flatten_portals(entrances)),
                abstract_graph_size=len(adjusted_graph),
                extra={},
            )

        refined_path = self._refine_path(abstract_path, start, goal)
        path_cost = self._compute_path_cost(refined_path)
        return SearchResult(
            path=refined_path,
            path_length=float(len(refined_path) - 1),
            path_cost=path_cost,
            nodes_expanded=nodes_expanded_total,
            execution_time_ms=elapsed_ms,
            preprocessing_time_ms=self._preprocessing_time_ms,
            preprocessing_nodes_expanded=self._preprocessing_nodes_expanded,
            success=True,
            num_portals=len(self._flatten_portals(entrances)),
            abstract_graph_size=len(adjusted_graph),
            extra={"abstract_nodes": list(abstract_path)},
        )

    def _refine_path(self, abstract_path: list[tuple[Any, ...]], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Refina um caminho abstrato em um caminho real usando A* entre nós consecutivos."""
        if not abstract_path:
            return [start]

        path: list[tuple[int, int]] = [start]
        for previous, current in zip(abstract_path, abstract_path[1:]):
            segment_start = self._node_to_point(previous, start, goal)
            segment_goal = self._node_to_point(current, start, goal)
            if segment_start is None or segment_goal is None:
                continue

            ordered_pair = (segment_start, segment_goal)
            reverse_pair = (segment_goal, segment_start)
            if ordered_pair in self._local_path_cost_cache:
                cached_path = self._local_path_cost_cache[ordered_pair]
            elif reverse_pair in self._local_path_cost_cache:
                cached_path = list(reversed(self._local_path_cost_cache[reverse_pair]))
            else:
                region = self._bounded_region_for_points(segment_start, segment_goal)
                local_grid = BoundedGridView(self.grid, *region)
                local_result = AStar(local_grid, max_nodes_expanded=self._local_astar_limit).search(segment_start, segment_goal)
                self._record_nodes_expanded(local_result.nodes_expanded)
                if local_result.success and local_result.path:
                    cached_path = list(local_result.path)
                    self._local_path_cost_cache[ordered_pair] = cached_path
                    self._local_path_cost_cache[reverse_pair] = list(reversed(cached_path))
                else:
                    cached_path = []
                    self._local_path_cost_cache[ordered_pair] = cached_path
                    self._local_path_cost_cache[reverse_pair] = cached_path

            if not cached_path:
                continue
            if len(cached_path) > 1:
                path.extend(cached_path[1:])
            else:
                path.append(segment_goal)

        if path[-1] != goal:
            region = self._bounded_region_for_points(path[-1], goal)
            local_grid = BoundedGridView(self.grid, *region)
            local_result = AStar(local_grid, max_nodes_expanded=self._local_astar_limit).search(path[-1], goal)
            self._record_nodes_expanded(local_result.nodes_expanded)
            if local_result.success:
                path.extend(local_result.path[1:])
        return self._dedupe_path(path)

    def _cluster_id_for_point(self, point: tuple[int, int]) -> int:
        """Retorna o identificador do cluster que contém ``point``."""
        if not self._clusters:
            self._build_clusters()
        return self._cluster_lookup.get(point, 0)

    def _bounded_region_for_points(self, start: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int, int, int]:
        """Determina a região de busca local para um par de pontos em clusters próximos."""
        first_cluster = self._cluster_id_for_point(start)
        second_cluster = self._cluster_id_for_point(goal)
        bounds_a = self._clusters[first_cluster]
        bounds_b = self._clusters[second_cluster]
        x0 = max(0, min(bounds_a[0], bounds_b[0]) - 2)
        y0 = max(0, min(bounds_a[1], bounds_b[1]) - 2)
        x1 = min(self.grid.width - 1, max(bounds_a[2], bounds_b[2]) + 2)
        y1 = min(self.grid.height - 1, max(bounds_a[3], bounds_b[3]) + 2)
        return x0, y0, x1, y1

    def _adjacent_cluster_ids(self, cluster_id: int) -> list[int]:
        """Retorna ids de clusters vizinhos em uma grade regular."""
        x0, y0, x1, y1 = self._clusters[cluster_id]
        neighbors: list[int] = []
        for other_id, other_bounds in enumerate(self._clusters):
            if other_id == cluster_id:
                continue
            ox0, oy0, ox1, oy1 = other_bounds
            if (x1 + 1 >= ox0 and ox1 >= x0 - 1 and y1 + 1 >= oy0 and oy1 >= y0 - 1):
                if abs(x0 - ox0) <= self.cluster_size or abs(y0 - oy0) <= self.cluster_size:
                    neighbors.append(other_id)
        return neighbors

    def _is_boundary_cell(self, point: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
        """Indica se a célula está na fronteira do cluster."""
        x, y = point
        x0, y0, x1, y1 = bounds
        return x in {x0, x1} or y in {y0, y1}

    def _collect_boundary_segment_representatives(self, cluster_id: int, bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
        """Agrupa células livres contíguas em segmentos de fronteira compartilhados com outro cluster.

        Para cada lado do cluster, as células da borda são agrupadas em corridas contíguas.
        Cada corrida vira uma única entrada representativa; segmentos longos usam os extremos
        para preservar melhor a conectividade sem reintroduzir uma entrada por célula.
        """
        x0, y0, x1, y1 = bounds
        representatives: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def has_neighbor_on_side(x: int, y: int, dx: int, dy: int) -> bool:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.grid.width and 0 <= ny < self.grid.height):
                return False
            return self._cluster_lookup.get((nx, ny), -1) != cluster_id

        def consume_segment(segment: list[tuple[int, int]]) -> None:
            """Escolhe no máximo 2 representantes por segmento (início e fim).

            Antes, segmentos com até 4 células mantinham TODAS as células como
            representantes -- com obstáculos espalhados, a maioria dos segmentos de
            fronteira acaba curta (fragmentada), então isso praticamente não reduzia
            o número de entradas (chegava a ~76 entradas por cluster em mapas
            realistas). Limitar a 1-2 pontos por segmento, sempre, é o que de fato
            mantém o grafo abstrato pequeno, preservando conectividade (cada run
            contíguo de células livres na fronteira ainda tem pelo menos 1 entrada).
            """
            if not segment:
                return
            if len(segment) == 1:
                points = segment
            else:
                points = [segment[0], segment[-1]]
            for point in points:
                if point not in seen:
                    seen.add(point)
                    representatives.append(point)

        def collect_side_points(start: tuple[int, int], step: tuple[int, int], side_check: tuple[int, int]) -> list[list[tuple[int, int]]]:
            """Varre um lado do cluster, limitado aos bounds DESTE cluster.

            Bug corrigido: a condição de parada usava os limites do GRID INTEIRO
            (self.grid.width/height) em vez dos limites do cluster (x0,y0,x1,y1).
            Isso fazia o scan "vazar" para muito além da fronteira do cluster,
            atravessando o mapa inteiro e capturando pontos de fronteira de
            clusters completamente distantes -- daí o número de entradas por
            cluster estar 10-20x maior do que deveria em mapas grandes.
            """
            segments: list[list[tuple[int, int]]] = []
            current: list[tuple[int, int]] = []
            cursor = start
            while x0 <= cursor[0] <= x1 and y0 <= cursor[1] <= y1:
                x, y = cursor
                point = (x, y)
                if self.grid.is_walkable(x, y) and has_neighbor_on_side(x, y, *side_check):
                    if current and point != (current[-1][0] + step[0], current[-1][1] + step[1]):
                        segments.append(current)
                        current = [point]
                    else:
                        current.append(point)
                else:
                    if current:
                        segments.append(current)
                        current = []
                cursor = (cursor[0] + step[0], cursor[1] + step[1])
            if current:
                segments.append(current)
            return segments

        for side in [
            ((x0, y0), (0, 1), (-1, 0)),
            ((x1, y0), (0, 1), (1, 0)),
            ((x0, y0), (1, 0), (0, -1)),
            ((x0, y1), (1, 0), (0, 1)),
        ]:
            start, step, side_check = side
            for segment in collect_side_points(start, step, side_check):
                consume_segment(segment)

        return representatives

    def _dedupe_points(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return list(dict.fromkeys(points))

    def _flatten_portals(self, entrances: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int]]:
        flattened: list[tuple[int, int]] = []
        for portal_list in entrances.values():
            flattened.extend(portal_list)
        return flattened

    def _record_nodes_expanded(self, count: int = 1) -> None:
        """Registra expansões de nós na métrica correta: preprocessamento ou consulta."""
        if self._active_metric_scope == "preprocessing":
            self._preprocessing_nodes_expanded += count
        elif self._active_metric_scope == "query":
            self._query_nodes_expanded += count
        self._last_nodes_expanded = self._query_nodes_expanded

    def _shortest_path(
        self,
        graph: dict[tuple[Any, ...], dict[tuple[Any, ...], float]],
        start: tuple[Any, ...],
        goal: tuple[Any, ...],
    ) -> list[tuple[Any, ...]]:
        """Busca de menor custo no grafo abstrato com fila de prioridade."""
        pq: PriorityQueue[tuple[Any, ...]] = PriorityQueue()
        pq.push(start, 0.0)
        best_cost: dict[tuple[Any, ...], float] = {start: 0.0}
        came_from: dict[tuple[Any, ...], tuple[Any, ...] | None] = {start: None}

        while not pq.is_empty():
            current = pq.pop()
            self._record_nodes_expanded()
            if current == goal:
                break
            for neighbor, weight in graph.get(current, {}).items():
                candidate = best_cost[current] + weight
                if candidate < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = candidate
                    came_from[neighbor] = current
                    pq.push(neighbor, candidate)

        if goal not in came_from:
            return []

        path: list[tuple[Any, ...]] = []
        cursor: tuple[Any, ...] | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = came_from[cursor]
        path.reverse()
        return path

    def _node_to_point(
        self,
        node: tuple[Any, ...],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Extrai a célula real de um nó abstrato, quando disponível."""
        if len(node) >= 3 and node[0] == "portal":
            return (int(node[2]), int(node[3]))
        if node[0] == "start":
            return start
        if node[0] == "goal":
            return goal
        return None

    def _dedupe_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Remove passos repetidos, preservando a ordem."""
        if not path:
            return []
        deduped = [path[0]]
        for point in path[1:]:
            if point != deduped[-1]:
                deduped.append(point)
        return deduped

    def _compute_path_cost(self, path: list[tuple[int, int]]) -> float:
        total = 0.0
        for previous, current in zip(path, path[1:]):
            total += self.grid.cost(previous, current)
        return total
