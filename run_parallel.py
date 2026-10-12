"""Configure and run src_parallel/main.jl.

Edit INPUT_FILES, OUTPUT_DIRECTORY, DATA_RANGE, and PARAMETERS below. The
launcher writes a temporary TOML configuration and starts one Julia worker for
each input file.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent
JULIA_SOURCE_DIRECTORY = REPOSITORY_ROOT / "src_parallel"

# Use absolute paths or paths relative to this Python file.
INPUT_FILES = [
    JULIA_SOURCE_DIRECTORY / "processed_trajectory_dynein_1.h5",
    JULIA_SOURCE_DIRECTORY / "processed_trajectory_dynein_2.h5",
    JULIA_SOURCE_DIRECTORY / "processed_trajectory_dynein_3.h5",
    JULIA_SOURCE_DIRECTORY / "processed_trajectory_dynein_4.h5",
]
OUTPUT_DIRECTORY = JULIA_SOURCE_DIRECTORY

# Use "all" for every observation, or a two-item tuple such as (1, 10_000).
# Julia indexing is one-based and both endpoints are included.
DATA_RANGE: str | tuple[int, int] = "all"

DT = 5.0e-7
TYPICAL_DIFFUSION_COEFFICIENT = 1.0e-13

# These defaults have the same values and Julia types as input_parameters.jl.
# Keep integer settings as Python ints and floating-point settings as floats.
PARAMETERS: dict[str, bool | int | float | str] = {
    # Availability of ground truth and known constants
    "ground_truth_available": False,
    "kb": 1.38e-23,
    "T": 300,
    "dt": DT,
    # Output parameters
    "trace_to_plot": 2,
    "output_frequency": 100,
    "plotting_on": True,
    "save_plots": False,
    "data_plotting_linewidth": 1.0,
    "data_plotting_opacity": 0.4,
    "data_plotting_sampling_frequency": 100,
    "n_samples_to_save": 200,
    # Bounds used for numerical stability
    "minimum_allowed_jump_size": 5.0e-9,
    "maximum_allowed_jump_size": 50.0e-9,
    # Prior parameters
    "typical_dwell_gamma_shape": 2.0,
    "typical_dwell_time": 3500 * 10 * DT,
    "dwell_gamma_shape_uniform_prior_min": 1.0,
    "dwell_gamma_shape_uniform_prior_max": 5.0,
    "dwell_gamma_mean_prior_shape": 1.0,
    "typical_jump_size": 8.0e-9,
    "std_jump_size_prior": 8.0e-9,
    "typical_k_value": 7.5e-5,
    "k_gamma_prior_shape": 1.0,
    "typical_diffusion_coeff": TYPICAL_DIFFUSION_COEFFICIENT,
    "std_diffusion_coeff_prior": 100.0 * TYPICAL_DIFFUSION_COEFFICIENT,
    # Model and nonparametric settings
    "modeling_choice": "nonparametric",
    "expected_n_dwells": 10,
    # Proposal step-size controls
    "conc_parameter_change_points_proposal": 1000.0,
    "step_size_position_sampler": 0.5e-9,
    "step_size_diffusion_coeff_sampler": 1.0e-2,
    "step_size_dwell_gamma": 5.0e-1,
    "stiffness_gamma_proposal_alpha": 5.0e2,
    "stiffness_xy_proposal_alpha": 5.0e2,
    # Simulated annealing and diagnostics
    "starting_temperature": 20000.0,
    "annealing_constant": 1000.0,
    "window_size": 100,
}

# The old code used seed 0 on the master and seeds 1..N on workers.
BASE_RANDOM_SEED = 0


def _toml_value(value: Any) -> str:
    """Serialize the small TOML value subset used by this launcher."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration floats must be finite")
        return repr(value)
    if isinstance(value, (str, Path)):
        # JSON basic strings use escaping compatible with TOML basic strings.
        return json.dumps(str(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _write_toml(path: Path, sections: dict[str, dict[str, Any]]) -> None:
    lines: list[str] = []
    for section_name, settings in sections.items():
        lines.append(f"[{section_name}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in settings.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    julia_executable = os.environ.get("JULIA_EXECUTABLE") or shutil.which("julia")
    if julia_executable is None:
        raise RuntimeError(
            "Julia was not found. Put it on PATH or set JULIA_EXECUTABLE to its path."
        )

    input_files = [
        path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()
        for path in map(Path, INPUT_FILES)
    ]
    output_directory = Path(OUTPUT_DIRECTORY)
    if not output_directory.is_absolute():
        output_directory = (REPOSITORY_ROOT / output_directory).resolve()

    configuration = {
        "run": {
            "input_files": [str(path) for path in input_files],
            "output_directory": str(output_directory),
            "base_random_seed": BASE_RANDOM_SEED,
            "data_range": DATA_RANGE,
        },
        "parameters": PARAMETERS,
    }

    with TemporaryDirectory(prefix="bnp-walk-") as temporary_directory:
        configuration_path = Path(temporary_directory) / "run.toml"
        _write_toml(configuration_path, configuration)

        command = [
            julia_executable,
            "--startup-file=no",
            str(JULIA_SOURCE_DIRECTORY / "main.jl"),
            str(configuration_path),
        ]
        subprocess.run(command, cwd=JULIA_SOURCE_DIRECTORY, check=True)


if __name__ == "__main__":
    main()
