"""Montagem de uma suíte de benchmark parametrizável para os algoritmos de pathfinding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

SIZES = [256, 512, 1024]
DENSITIES = {"low": 0.10, "medium": 0.25, "high": 0.40}
MAPS_PER_CONFIGURATION = 10
SCENARIOS_PER_MAP = 50
REPETITIONS = 30


def _discover_real_maps(maps_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Descobre mapas reais .map e seus cenários .scen correspondentes em data/maps."""
    root = Path(maps_dir or Path(__file__).resolve().parents[2] / "data" / "maps")
    if not root.exists():
        return []

    map_files = sorted(root.glob("*.map"))
    if not map_files:
        return []

    discovered: list[dict[str, Any]] = []
    for map_file in map_files:
        scen_files = sorted(root.glob(f"{map_file.stem}-random-*.scen"))
        if not scen_files:
            scen_files = [root / f"{map_file.stem}.scen"] if (root / f"{map_file.stem}.scen").exists() else []
        discovered.append(
            {
                "map_path": str(map_file),
                "map_name": map_file.stem,
                "scen_paths": [str(path) for path in scen_files],
            }
        )
    return discovered


def build_benchmark_suite(
    sizes: Sequence[int] | None = None,
    densities: Mapping[str, float] | None = None,
    maps_per_configuration: int = MAPS_PER_CONFIGURATION,
    scenarios_per_map: int = SCENARIOS_PER_MAP,
    repetitions: int = REPETITIONS,
    use_real_maps: bool = False,
) -> list[dict[str, Any]]:
    """Cria a lista de experimentos sem executá-los."""
    selected_sizes = list(sizes or SIZES)
    selected_densities = dict(densities or DENSITIES)
    suite: list[dict[str, Any]] = []

    real_maps = _discover_real_maps()
    if real_maps and use_real_maps:
        for map_info in real_maps:
            suite.append(
                {
                    "size": None,
                    "width": None,
                    "height": None,
                    "density": 0.0,
                    "density_label": "real",
                    "map_index": 0,
                    "map_name": map_info["map_name"],
                    "map_path": map_info["map_path"],
                    "scen_paths": map_info["scen_paths"],
                    "scenarios": [],
                    "repetitions": repetitions,
                    "source": "real_map",
                }
            )
        return suite

    for size in selected_sizes:
        for density_label, density_value in selected_densities.items():
            for map_index in range(maps_per_configuration):
                scenarios = []
                for scenario_index in range(scenarios_per_map):
                    start = (
                        scenario_index % max(2, size // 64),
                        (scenario_index * 7) % max(2, size // 64),
                    )
                    goal = (
                        size - 1 - ((scenario_index + 3) % max(2, size // 64)),
                        size - 1 - ((scenario_index * 11) % max(2, size // 64)),
                    )
                    scenarios.append(
                        {
                            "scenario": scenario_index,
                            "start": start,
                            "goal": goal,
                        }
                    )
                suite.append(
                    {
                        "size": size,
                        "width": size,
                        "height": size,
                        "density": density_value,
                        "density_label": density_label,
                        "map_index": map_index,
                        "map_name": f"synthetic_{size}_{density_label}_{map_index}",
                        "scenarios": scenarios,
                        "repetitions": repetitions,
                    }
                )

    return suite
