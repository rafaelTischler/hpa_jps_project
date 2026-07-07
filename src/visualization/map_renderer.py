"""Renderização simples de mapas com caminho e nós de fronteira."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from src.core.grid import Grid


def render_map(
    grid: Grid,
    output_path: str | Path = "results/plots/map_render.png",
    path: Sequence[tuple[int, int]] | None = None,
    portals: Sequence[tuple[int, int]] | None = None,
    jump_points: Sequence[tuple[int, int]] | None = None,
    title: str = "Grid",
) -> str:
    """Desenha o grid com obstáculos, caminho e pontos de entrada em destaque."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for y in range(grid.height):
        for x in range(grid.width):
            color = "white" if grid.is_walkable(x, y) else "black"
            ax.add_patch(plt.Rectangle((x, grid.height - y - 1), 1, 1, facecolor=color, edgecolor="0.7"))

    if path:
        for x, y in path:
            ax.add_patch(plt.Rectangle((x, grid.height - y - 1), 1, 1, facecolor="red", alpha=0.6, edgecolor="none"))

    if portals:
        for x, y in portals:
            ax.add_patch(plt.Rectangle((x, grid.height - y - 1), 1, 1, facecolor="blue", alpha=0.6, edgecolor="none"))

    if jump_points:
        for x, y in jump_points:
            ax.add_patch(plt.Rectangle((x, grid.height - y - 1), 1, 1, facecolor="green", alpha=0.6, edgecolor="none"))

    ax.set_title(title)
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)
