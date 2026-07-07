"""Carregamento de mapas e cenários do benchmark movingai.com."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from src.core.grid import Grid

# Caracteres transitáveis e bloqueados conforme SPEC.md / movingai.com.
_WALKABLE_CHARS = frozenset({".", "G"})
_OBSTACLE_CHARS = frozenset({"@", "O", "T", "W"})


@dataclass(frozen=True)
class Scenario:
    """Par start/goal extraído de um arquivo ``.scen``."""

    bucket: int
    map_name: str
    map_width: int
    map_height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    optimal_length: float


def parse_map_content(content: str) -> Grid:
    """Faz parse de conteúdo textual no formato ``.map`` do movingai.

    Espera cabeçalho ``type octile``, ``height``, ``width``, ``map`` e a matriz
    ASCII (``.``/``G`` transitáveis; ``@``/``O``/``T``/``W`` obstáculos).

    Args:
        content: Texto completo do arquivo ``.map``.

    Returns:
        Instância de ``Grid`` pronta para os algoritmos.

    Raises:
        ValueError: Se o cabeçalho ou a matriz forem inválidos.
    """
    lines = [line.rstrip("\r\n") for line in content.strip().splitlines()]
    if not lines:
        raise ValueError("empty map content")

    if lines[0].strip().lower() != "type octile":
        raise ValueError(f"expected 'type octile' header, got {lines[0]!r}")

    height: int | None = None
    width: int | None = None
    map_start_idx = 0

    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("height"):
            height = int(stripped.split()[1])
        elif lower.startswith("width"):
            width = int(stripped.split()[1])
        elif lower == "map":
            map_start_idx = idx + 1
            break

    if height is None or width is None:
        raise ValueError("map header must include height and width")
    if map_start_idx == 0:
        raise ValueError("map header must include 'map' marker")

    map_lines = lines[map_start_idx : map_start_idx + height]
    if len(map_lines) != height:
        raise ValueError(
            f"expected {height} map rows, found {len(map_lines)}"
        )

    walkable: list[list[bool]] = []
    for row_idx, row in enumerate(map_lines):
        if len(row) != width:
            raise ValueError(
                f"row {row_idx} has width {len(row)}, expected {width}"
            )
        walkable_row: list[bool] = []
        for char in row:
            if char in _WALKABLE_CHARS:
                walkable_row.append(True)
            elif char in _OBSTACLE_CHARS:
                walkable_row.append(False)
            else:
                raise ValueError(f"unknown map character {char!r}")
        walkable.append(walkable_row)

    return Grid(walkable)


def load_map(path: str | Path) -> Grid:
    """Carrega um arquivo ``.map`` do disco e retorna um ``Grid``."""
    map_path = Path(path)
    content = map_path.read_text(encoding="utf-8")
    return parse_map_content(content)


def parse_scen_content(content: str) -> list[Scenario]:
    """Faz parse de conteúdo textual no formato ``.scen`` do movingai.

    Cada linha não vazia contém 9 campos: bucket, mapa, largura, altura,
    start_x, start_y, goal_x, goal_y, comprimento ótimo.

    Args:
        content: Texto completo do arquivo ``.scen``.

    Returns:
        Lista de cenários parseados.
    """
    scenarios: list[Scenario] = []
    for line in content.strip().splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("version"):
            continue

        parts = stripped.split()
        if len(parts) < 9:
            raise ValueError(f"invalid scen line (expected 9 fields): {stripped!r}")

        bucket = int(parts[0])
        map_name = parts[1]
        map_width = int(parts[2])
        map_height = int(parts[3])
        start = (int(parts[4]), int(parts[5]))
        goal = (int(parts[6]), int(parts[7]))
        optimal_length = float(parts[8])

        scenarios.append(
            Scenario(
                bucket=bucket,
                map_name=map_name,
                map_width=map_width,
                map_height=map_height,
                start=start,
                goal=goal,
                optimal_length=optimal_length,
            )
        )

    return scenarios


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Carrega cenários de um arquivo ``.scen`` do disco."""
    scen_path = Path(path)
    content = scen_path.read_text(encoding="utf-8")
    return parse_scen_content(content)


def generate_synthetic_map(
    width: int,
    height: int,
    density: float,
    seed: int | None = 42,
) -> Grid:
    """Gera mapa sintético com densidade controlada de obstáculos.

    ``density`` é a fração de células bloqueadas (ex.: 0.10 ≈ baixa,
    0.25 ≈ média, 0.40 ≈ alta). Usa ``seed`` fixa por padrão para
    reprodutibilidade nos experimentos.

    Args:
        width: Largura do grid.
        height: Altura do grid.
        density: Proporção de obstáculos em [0, 1].
        seed: Semente do gerador pseudo-aleatório (None = não determinístico).

    Returns:
        ``Grid`` sintético com obstáculos espalhados aleatoriamente.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be between 0 and 1")

    rng = random.Random(seed)
    walkable: list[list[bool]] = []
    for _ in range(height):
        row = [rng.random() >= density for _ in range(width)]
        walkable.append(row)

    return Grid(walkable)


def parse_map_from_string(content: str) -> Grid:
    """Alias conveniente para testes — equivalente a ``parse_map_content``."""
    return parse_map_content(content)
