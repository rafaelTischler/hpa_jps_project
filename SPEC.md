# PROMPT DE IMPLEMENTAÇÃO — Projeto HPA-JPS (Python)

Você é um engenheiro de software sênior especialista em algoritmos de pathfinding e em
Python. Implemente, do zero, um projeto completo em Python 3.11+ para o trabalho acadêmico
"Integração entre HPA* e Jump Point Search". Siga EXATAMENTE a arquitetura, os contratos de
classe e a ordem de implementação descritos abaixo. Não pule etapas. Ao final, o projeto deve
rodar localmente (VSCode), sem dependências pagas, e gerar automaticamente os dados/gráficos
necessários para o relatório técnico.

## 0. Objetivo geral

Implementar e comparar 4 algoritmos de busca de caminho em grids: A*, JPS, HPA* e uma
proposta de integração HPA-JPS (Jump Points substituindo os portais tradicionais do HPA*),
avaliando-os em mapas do benchmark movingai.com, com diferentes tamanhos e densidades de
obstáculos, e registrando métricas de desempenho e qualidade de caminho.

## 1. Estrutura de pastas do projeto

```
hpa_jps_project/
├── README.md
├── requirements.txt
├── main.py                          # CLI de entrada
├── data/
│   └── maps/                        # arquivos .map / .scen baixados do movingai
├── results/
│   ├── raw/                         # CSVs brutos de cada execução
│   └── plots/                       # gráficos gerados (PNG)
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── grid.py                  # Grid, Node, custo, vizinhança
│   │   ├── heuristics.py            # octile, euclidiana, manhattan
│   │   ├── priority_queue.py        # wrapper de heapq (fila de prioridade reutilizável)
│   │   └── result.py                # dataclass SearchResult (métricas padronizadas)
│   ├── io/
│   │   ├── __init__.py
│   │   └── map_loader.py            # parser dos formatos .map e .scen do movingai
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_pathfinder.py       # classe abstrata PathfindingAlgorithm
│   │   ├── astar.py                 # AStar(PathfindingAlgorithm)
│   │   ├── jps.py                   # JPS(AStar) -> reaproveita fila, heurística, reconstrução
│   │   ├── hpa_star.py              # HPAStar(PathfindingAlgorithm) -> reaproveita AStar internamente
│   │   └── hpa_jps.py               # HPAJPS(HPAStar) -> reaproveita HPAStar + JPS.identify_jump_points
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── experiment_runner.py     # orquestra execuções em lote
│   │   └── metrics_writer.py        # grava CSV com as métricas exigidas
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plotter.py               # gráficos comparativos (matplotlib)
│   │   └── map_renderer.py          # desenha grid + caminho + portais/jump points (debug)
│   └── utils/
│       ├── __init__.py
│       └── timing.py                # decorator de cronometragem reutilizável
└── tests/
    ├── __init__.py
    ├── test_astar.py
    ├── test_jps.py
    ├── test_hpa_star.py
    └── test_hpa_jps.py
```

## 2. Princípio de reaproveitamento (obrigatório)

Nenhum algoritmo deve reimplementar o que outro já resolve. Regras de herança/composição:

- `AStar` implementa a busca de menor caminho genérica em grid usando `Grid`, `PriorityQueue`
  e `heuristics`. Essa é a base de tudo.
- `JPS` **herda de `AStar`** e sobrescreve apenas a função de expansão de vizinhos
  (`get_successors`), substituindo-a pela lógica de "jumping" (pruning de vizinhos + busca
  recursiva de jump points). Reaproveita de `AStar`: fila de prioridade, cálculo de g/h/f,
  reconstrução de caminho, contagem de nós expandidos, cronometragem.
- `HPAStar` usa **composição** com uma instância interna de `AStar` para (a) resolver buscas
  locais dentro de cada cluster e (b) resolver a busca no grafo abstrato de clusters/portais.
  Não reimplementa A* — apenas chama `AStar.search()` internamente.
- `HPAJPS` **herda de `HPAStar`** e sobrescreve apenas o método responsável por gerar os nós
  de fronteira entre clusters (`build_abstract_graph` / `_generate_entrances`), delegando a
  identificação desses nós para um método estático de `JPS`
  (`JPS.identify_jump_points(grid, cluster_bounds)`) em vez de usar portais geométricos fixos.
  Todo o resto (clusterização, cache de caminhos intra-cluster, busca hierárquica, refinamento
  do caminho abstrato em caminho real) é herdado de `HPAStar` sem modificação.

Isso garante: código validado uma vez (em `AStar`/`HPAStar`) é reaproveitado, e cada algoritmo
novo adiciona só o que é conceitualmente diferente.

## 3. Contratos de classe (implementar exatamente essas interfaces)

### 3.1 `src/core/result.py`
```python
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    path: list[tuple[int, int]]
    path_length: float          # número de passos ou distância euclidiana acumulada
    path_cost: float            # custo total ponderado (considerando diagonais, se houver)
    nodes_expanded: int
    execution_time_ms: float
    success: bool
    # campos opcionais, preenchidos apenas por algoritmos hierárquicos:
    num_portals: int = 0
    num_jump_points: int = 0
    abstract_graph_size: int = 0
    extra: dict = field(default_factory=dict)
```

### 3.2 `src/algorithms/base_pathfinder.py`
```python
from abc import ABC, abstractmethod
from src.core.grid import Grid
from src.core.result import SearchResult

class PathfindingAlgorithm(ABC):
    def __init__(self, grid: Grid):
        self.grid = grid

    @abstractmethod
    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> SearchResult:
        ...

    def name(self) -> str:
        return self.__class__.__name__
```

### 3.3 `src/core/grid.py`
- Classe `Grid`: carrega matriz booleana de células transitáveis/obstáculo, expõe:
  - `is_walkable(x, y) -> bool`
  - `neighbors(x, y, allow_diagonal=True) -> list[tuple[int,int]]` (com checagem de "corner
    cutting" — bloquear diagonal se ambos os ortogonais adjacentes forem obstáculo)
  - `cost(a, b) -> float` (1.0 ortogonal, sqrt(2) diagonal)
  - `width`, `height`

### 3.4 `src/core/heuristics.py`
- `octile(a, b)`, `euclidean(a, b)`, `manhattan(a, b)` — funções puras reutilizadas por
  `AStar`, `JPS` e nas buscas internas de `HPAStar`.

### 3.5 `src/algorithms/astar.py`
- `class AStar(PathfindingAlgorithm)`
- Método `search(start, goal)`: A* padrão com fila de prioridade binária, heurística octile
  por padrão (parametrizável), fecha em `SearchResult`.
- Método protegido `get_successors(node)` retornando vizinhos — **este é o método que `JPS`
  vai sobrescrever**.
- Contar `nodes_expanded` incrementando a cada pop da fila.

### 3.6 `src/algorithms/jps.py`
- `class JPS(AStar)`
- Sobrescreve `get_successors` implementando:
  - poda de vizinhos (natural neighbors + forced neighbors) conforme Harabor & Grastien (2012,
    2014);
  - função recursiva `_jump(node, direction, start, goal)` que avança na direção até encontrar
    um jump point, obstáculo, ou borda do mapa.
- Método estático **reaproveitável por `HPAJPS`**:
  `@staticmethod identify_jump_points(grid: Grid, region: tuple[int,int,int,int]) -> list[tuple[int,int]]`
  — varre a sub-região do grid e retorna todos os jump points estruturais (pontos com vizinho
  forçado), sem calcular caminho algum. Essa função é o ponto de integração com HPA-JPS.

### 3.7 `src/algorithms/hpa_star.py`
- `class HPAStar(PathfindingAlgorithm)`
- Parâmetro `cluster_size` (ex.: 10).
- Etapas obrigatórias, cada uma em seu próprio método (para permitir override em `HPAJPS`):
  1. `_build_clusters()` — particiona o grid em clusters `cluster_size x cluster_size`.
  2. `_generate_entrances()` — **método a ser sobrescrito por `HPAJPS`**. Na versão base,
     gera portais geometricamente nas fronteiras entre clusters adjacentes (implementação
     clássica de Botea et al., 2004 / Jansen & Buro, 2007).
  3. `_build_abstract_graph()` — cria grafo (networkx ou estrutura própria) ligando as
     entradas/portais entre clusters vizinhos e dentro do mesmo cluster (usando `AStar`
     internamente para achar o custo intra-cluster entre cada par de entradas do cluster —
     **aqui é onde `AStar` é reaproveitado, nunca reimplementado**).
  4. `search(start, goal)`:
     - insere start/goal no grafo abstrato (conectando aos nós de entrada do cluster local via
       `AStar`);
     - roda A* (ou Dijkstra) no grafo abstrato;
     - refina (`_refine_path`) o caminho abstrato em caminho real célula-a-célula usando
       `AStar` local em cada segmento entre entradas.
  5. Preencher em `SearchResult`: `num_portals`, `abstract_graph_size`.

### 3.8 `src/algorithms/hpa_jps.py`
- `class HPAJPS(HPAStar)`
- Sobrescreve **apenas** `_generate_entrances()`:
  - para cada cluster, chama `JPS.identify_jump_points(self.grid, cluster_bounds)`;
  - opcionalmente também mantém portais de fronteira quando não há jump point suficiente na
    fronteira (estratégia "grafo híbrido" — deixe isso como flag `hybrid: bool` no
    construtor, default `True`, para permitir o desafio extra com `hybrid=False`, ou seja,
    substituição total dos portais).
- Reaproveita `_build_abstract_graph`, `search`, `_refine_path` de `HPAStar` sem alterações.
- Preenche também `num_jump_points` em `SearchResult`.

## 4. Ingestão de mapas (`src/io/map_loader.py`)

- Implementar parser do formato `.map` do movingai benchmarks (header `type octile`,
  `height`, `width`, `map`, seguido da matriz de caracteres `.` `G` transitáveis e `@` `O`
  `T` `W` obstáculos — ver documentação em https://www.movingai.com/benchmarks/formats.html).
- Implementar parser do formato `.scen` (arquivo de cenários com pares start/goal
  pré-definidos), para reproduzir experimentos padronizados da literatura.
- Função utilitária `generate_synthetic_map(width, height, density) -> Grid` para os casos em
  que se quiser controlar densidade exata (baixa ~10%, média ~25%, alta ~40% de obstáculos)
  além dos mapas baixados.

## 5. Benchmark e métricas (`src/benchmark/`)

- `experiment_runner.py`:
  - Função `run_experiment(map_path, algorithms: list[PathfindingAlgorithm], scenarios)` que
    roda todos os algoritmos sobre os mesmos pares start/goal e mesmo mapa.
  - Deve iterar sobre: tamanhos {256x256, 512x512, 1024x1024} × densidades {baixa, média,
    alta} × algoritmos {A*, JPS, HPA*, HPA-JPS} × N pares start/goal (ex.: 20 por
    combinação, para significância estatística).
- `metrics_writer.py`: grava um CSV (`results/raw/experiment_results.csv`) com colunas:
  `map_name, size, density, algorithm, run_id, execution_time_ms, nodes_expanded,
  path_length, path_cost, num_portals, num_jump_points, abstract_graph_size, success`.

## 6. Visualização (`src/visualization/plotter.py`)

Gerar automaticamente (salvando em `results/plots/`), a partir do CSV:
1. Gráfico de barras: tempo médio de execução por algoritmo, agrupado por tamanho de mapa.
2. Gráfico de barras: nós expandidos por algoritmo.
3. Gráfico de dispersão/linha: tamanho do grafo abstrato (HPA* vs HPA-JPS) por densidade.
4. Gráfico comparando qualidade do caminho (custo/comprimento) entre os 4 algoritmos, para
   checar se HPA-JPS preserva otimalidade aproximada.
5. Tabela resumo (exportar também como `.csv` e como Markdown) pronta para colar no relatório.

`map_renderer.py` (opcional, para o vídeo/relatório): desenhar o grid com obstáculos, caminho
encontrado, e destacar portais (HPA*) vs jump points (HPA-JPS) em cores diferentes.

## 7. CLI (`main.py`)

Implementar com `argparse`, comandos:
```
python main.py run --map data/maps/arena.map --algorithms astar jps hpastar hpajps
python main.py experiment --sizes 256 512 1024 --densities low medium high
python main.py plot --input results/raw/experiment_results.csv
python main.py render --map data/maps/arena.map --algorithm hpajps --start 0,0 --goal 50,50
```

## 8. Testes (`tests/`)

- Casos pequenos (grid 10x10 com obstáculos conhecidos, caminho ótimo calculado manualmente)
  garantindo que os 4 algoritmos retornam o mesmo `path_cost` (permitindo tolerância de ponto
  flutuante) — isso valida que a hierarquia (HPA*/HPA-JPS) não perde otimalidade indevidamente.
- Teste específico de `JPS.identify_jump_points` isolado.
- Teste de regressão comparando `nodes_expanded` de JPS vs A* no mesmo mapa (JPS deve expandir
  igual ou menos).

## 9. requirements.txt

```
numpy
matplotlib
networkx
pandas
pytest
```

## 10. README.md a gerar

Deve conter: como instalar (`pip install -r requirements.txt`), como baixar mapas do
movingai.com e colocá-los em `data/maps/`, como rodar os experimentos completos, como gerar
os gráficos, e como rodar os testes (`pytest tests/`).

## 11. Ordem de implementação sugerida (siga esta sequência)

1. `core/grid.py`, `core/heuristics.py`, `core/priority_queue.py`, `core/result.py`
2. `algorithms/base_pathfinder.py`
3. `algorithms/astar.py` + `tests/test_astar.py`
4. `algorithms/jps.py` + `tests/test_jps.py`
5. `io/map_loader.py`
6. `algorithms/hpa_star.py` + `tests/test_hpa_star.py`
7. `algorithms/hpa_jps.py` + `tests/test_hpa_jps.py`
8. `benchmark/experiment_runner.py` + `benchmark/metrics_writer.py`
9. `visualization/plotter.py` + `visualization/map_renderer.py`
10. `main.py` (CLI) + `README.md`

## 12. Critérios de aceitação

- Os 4 algoritmos implementam a mesma interface `PathfindingAlgorithm.search()` e retornam
  `SearchResult` preenchido corretamente.
- `HPAJPS` não duplica código de `HPAStar`; apenas sobrescreve `_generate_entrances`.
- `JPS` não duplica código de `AStar`; apenas sobrescreve `get_successors` (mais os métodos
  auxiliares de jump/poda).
- O CSV de resultados contém todas as métricas exigidas pelo enunciado (tempo, nós expandidos,
  comprimento, custo, número de portais, número de jump points, tamanho do grafo abstrato).
- Os gráficos gerados respondem visualmente às 5 perguntas de pesquisa do enunciado (seção 4
  do documento original).
- Todo código com docstrings citando a fonte teórica (Botea et al. 2004; Jansen & Buro 2007;
  Harabor & Grastien 2012, 2014) nos trechos que implementam ideias desses artigos.

---

**Instrução final para a IA que for implementar:** gere o código completo arquivo por
arquivo, na ordem da seção 11, mostrando o conteúdo integral de cada arquivo antes de passar
para o próximo. Ao final, gere também um exemplo de execução ponta a ponta (mapa pequeno
sintético) mostrando que os 4 algoritmos rodam e produzem `SearchResult` coerentes entre si.
