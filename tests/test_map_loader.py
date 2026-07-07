"""Testes unitários para carregamento de mapas e cenários."""

from __future__ import annotations

import pytest

from src.benchmark.experiment_runner import _generate_connected_synthetic_map
from src.io.map_loader import (
    generate_synthetic_map,
    parse_map_content,
    parse_scen_content,
)


SAMPLE_MAP = """\
type octile
height 4
width 5
map
.....
..@..
..G..
.....
"""

SAMPLE_SCEN = """\
version 1
0 test.map 5 4 0 0 4 3 5.0
1 test.map 5 4 1 1 3 2 3.41421
"""


class TestParseMap:
    """Parser do formato ``.map`` do movingai."""

    def test_dimensions_and_cells(self) -> None:
        grid = parse_map_content(SAMPLE_MAP)

        assert grid.width == 5
        assert grid.height == 4
        assert grid.is_walkable(0, 0) is True
        assert grid.is_walkable(2, 1) is False  # '@' obstáculo
        assert grid.is_walkable(2, 2) is True   # 'G' transitável
        assert grid.is_walkable(4, 3) is True


class TestParseScen:
    """Parser do formato ``.scen`` do movingai."""

    def test_start_goal_pairs(self) -> None:
        scenarios = parse_scen_content(SAMPLE_SCEN)

        assert len(scenarios) == 2

        first = scenarios[0]
        assert first.bucket == 0
        assert first.map_name == "test.map"
        assert first.map_width == 5
        assert first.map_height == 4
        assert first.start == (0, 0)
        assert first.goal == (4, 3)
        assert first.optimal_length == pytest.approx(5.0)

        second = scenarios[1]
        assert second.start == (1, 1)
        assert second.goal == (3, 2)
        assert second.optimal_length == pytest.approx(3.41421)


class TestGenerateSyntheticMap:
    """Geração de mapas sintéticos com densidade controlada."""

    def test_density_near_target(self) -> None:
        width, height = 100, 100
        density = 0.25
        grid = generate_synthetic_map(width, height, density, seed=123)

        assert grid.width == width
        assert grid.height == height

        total_cells = width * height
        obstacle_count = sum(
            1 for y in range(height) for x in range(width) if not grid.is_walkable(x, y)
        )
        actual_density = obstacle_count / total_cells

        # Tolerância de ±5 pontos percentuais para flutuação amostral.
        assert actual_density == pytest.approx(density, abs=0.05)

    def test_reproducible_with_seed(self) -> None:
        grid_a = generate_synthetic_map(20, 20, 0.3, seed=99)
        grid_b = generate_synthetic_map(20, 20, 0.3, seed=99)

        for y in range(20):
            for x in range(20):
                assert grid_a.is_walkable(x, y) == grid_b.is_walkable(x, y)

    def test_connected_synthetic_map_for_dense_case(self) -> None:
        grid = _generate_connected_synthetic_map(16, 16, 0.4, seed=42)

        assert grid.is_walkable(0, 0) is True
        assert grid.is_walkable(15, 15) is True
