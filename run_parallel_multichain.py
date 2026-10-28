"""Run several trajectories concurrently with internally parallel MCMC chains.

Edit DEFAULT_PARAMETERS and TRAJECTORIES below. Each trajectory is handled by
one Julia instance running ``src_parallel_multichain/main.jl``. That Julia
instance starts ``n_MCMC_chains`` worker processes for the trajectory, while
this launcher runs up to MAX_PARALLEL_TRAJECTORIES Julia instances at once.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent
JULIA_SOURCE_DIRECTORY = REPOSITORY_ROOT / "src_parallel_multichain"
INPUT_DIRECTORY = REPOSITORY_ROOT / "src_parallel"
OUTPUT_DIRECTORY = JULIA_SOURCE_DIRECTORY

# This limits concurrent trajectories, not the chains inside each trajectory.
# With the defaults below, 2 concurrent trajectories use 2 Julia coordinators
# and 50 Julia MCMC worker processes.
MAX_PARALLEL_TRAJECTORIES = 2
PROGRESS_REPORT_INTERVAL_SECONDS = 30.0

DT = 5.0e-7
TYPICAL_DIFFUSION_COEFFICIENT = 1.0e-13

# These defaults have the same values and Julia types as the original
# src_parallel_multichain/input_parameters.jl. Keep integer settings as Python
# ints and floating-point settings as floats; the launcher validates both.
DEFAULT_PARAMETERS: dict[str, bool | int | float | str] = {
    # Availability of ground truth and known constants
    "ground_truth_available": False,
    "kb": 1.38e-23,
    "T": 300,
    "dt": DT,
    # Output parameters
    "chain_to_plot": 1,
    "output_frequency": 100,
    "plotting_on": True,
    "save_plots": False,
    "data_plotting_linewidth": 1.0,
    "data_plotting_opacity": 0.4,
    "data_plotting_sampling_frequency": 100,
    # Total samples across all chains; must be divisible by n_MCMC_chains.
    "n_MCMC_chains": 25,
    "n_samples_to_save": 100,
    # Bounds used for numerical stability
    "minimum_allowed_jump_size": 2.0e-9,
    "maximum_allowed_jump_size": 50.0e-9,
    # Prior parameters
    "typical_dwell_gamma_shape": 2.0,
    "typical_dwell_time": 20.0e-3,
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

# Each entry inherits DEFAULT_PARAMETERS and may override any parameter. Use
# "all" for the complete trajectory or a two-item tuple such as (1, 10_000)
# for data_range. Julia indexing is one-based and both endpoints are included.
#
# output_directory is optional and defaults to OUTPUT_DIRECTORY. Relative paths
# are resolved from this Python file's directory.
TRAJECTORIES: list[dict[str, Any]] = [
    {
        "input_file": INPUT_DIRECTORY / "processed_trajectory_dynein_1.h5",
        "data_range": "all",
        "parameters": {},
    },
    {
        "input_file": INPUT_DIRECTORY / "processed_trajectory_dynein_2.h5",
        "data_range": "all",
        "parameters": {
            # Example per-trajectory overrides:
            # "n_MCMC_chains": 10,
            # "n_samples_to_save": 100,
            # "expected_n_dwells": 15,
            # "typical_dwell_time": 0.025,
        },
    },
    {
        "input_file": INPUT_DIRECTORY / "processed_trajectory_dynein_3.h5",
        "data_range": "all",
        "parameters": {},
    },
    {
        "input_file": INPUT_DIRECTORY / "processed_trajectory_dynein_4.h5",
        "data_range": "all",
        "parameters": {},
    },
]

# The launcher assigns a disjoint seed range to every trajectory. A trajectory
# with N chains uses one coordinator seed and N worker seeds.
BASE_RANDOM_SEED = 0

_DRAW_PATTERN = re.compile(r"\bdraw = (\d+)\b")
_SAVED_SAMPLES_PATTERN = re.compile(r"Saved Samples upto No\. (\d+)\b")
_TERMINAL_LOCK = threading.Lock()


@dataclass
class _JobProgress:
    total_draws: int
    n_mcmc_chains: int
    completed_draws: int = 0
    status: str = "queued"
    started_at: float | None = None
    progress_baseline_draws: int | None = None
    progress_baseline_at: float | None = None


def _terminal_print(message: str, *, end: str = "\n") -> None:
    with _TERMINAL_LOCK:
        print(message, end=end, flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ProgressTracker:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self._started_at = time.monotonic()
        self._lock = threading.Lock()
        self._states = [
            _JobProgress(
                total_draws=job["total_draws"],
                n_mcmc_chains=job["n_mcmc_chains"],
            )
            for job in jobs
        ]

    def start(self, job_index: int) -> None:
        with self._lock:
            state = self._states[job_index]
            state.status = "running"
            state.started_at = time.monotonic()

    def update(self, job_index: int, completed_draws: int) -> None:
        with self._lock:
            state = self._states[job_index]
            updated_draws = min(
                state.total_draws,
                max(state.completed_draws, completed_draws),
            )
            if (
                updated_draws > state.completed_draws
                and state.progress_baseline_at is None
            ):
                state.progress_baseline_draws = updated_draws
                state.progress_baseline_at = time.monotonic()
            state.completed_draws = updated_draws

    def finish(self, job_index: int) -> None:
        with self._lock:
            state = self._states[job_index]
            state.completed_draws = state.total_draws
            state.status = "done"

    def fail(self, job_index: int) -> None:
        with self._lock:
            self._states[job_index].status = "failed"

    def report(self) -> None:
        now = time.monotonic()
        with self._lock:
            total_draws = sum(state.total_draws for state in self._states)
            completed_draws = sum(
                state.completed_draws for state in self._states
            )
            counts = {
                status: sum(state.status == status for state in self._states)
                for status in ("queued", "running", "done", "failed")
            }

            active_details: list[str] = []
            aggregate_rate = 0.0
            for job_index, state in enumerate(self._states):
                if state.status != "running" or state.started_at is None:
                    continue

                if (
                    state.progress_baseline_at is not None
                    and state.progress_baseline_draws is not None
                    and state.completed_draws > state.progress_baseline_draws
                ):
                    sampling_elapsed = max(
                        now - state.progress_baseline_at,
                        1.0e-9,
                    )
                    rate = (
                        state.completed_draws - state.progress_baseline_draws
                    ) / sampling_elapsed
                else:
                    rate = 0.0
                aggregate_rate += rate
                remaining = state.total_draws - state.completed_draws
                eta = remaining / rate if rate > 0.0 else None
                percent = 100.0 * state.completed_draws / state.total_draws
                active_details.append(
                    f"T{job_index + 1} {percent:5.1f}% "
                    f"({state.completed_draws:,}/{state.total_draws:,} draws, "
                    f"{state.n_mcmc_chains} chains) "
                    f"ETA {_format_duration(eta)}"
                )

            remaining_draws = total_draws - completed_draws
            if remaining_draws == 0:
                overall_eta = 0.0
            elif aggregate_rate > 0.0 and counts["failed"] == 0:
                overall_eta = remaining_draws / aggregate_rate
            else:
                overall_eta = None
            overall_percent = 100.0 * completed_draws / total_draws
            elapsed = now - self._started_at

        _terminal_print(
            f"[progress] overall {overall_percent:5.1f}% "
            f"({completed_draws:,}/{total_draws:,} synchronized draws); "
            f"running={counts['running']} queued={counts['queued']} "
            f"done={counts['done']} failed={counts['failed']}; "
            f"elapsed {_format_duration(elapsed)}; "
            f"ETA {_format_duration(overall_eta)}"
        )
        if active_details:
            _terminal_print("[progress] active: " + " | ".join(active_details))


def _report_progress_until_stopped(
    progress: _ProgressTracker,
    stop_event: threading.Event,
) -> None:
    progress.report()
    while not stop_event.wait(PROGRESS_REPORT_INTERVAL_SECONDS):
        progress.report()


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
        lines.extend(
            f"{key} = {_toml_value(value)}" for key, value in settings.items()
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _validate_parameter_types(parameters: dict[str, Any]) -> None:
    for name, default_value in DEFAULT_PARAMETERS.items():
        value = parameters[name]
        expected_type = type(default_value)
        if type(value) is not expected_type:
            raise TypeError(
                f"Parameter '{name}' must be {expected_type.__name__}, "
                f"not {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Parameter '{name}' must be finite")


def _validate_data_range(value: Any) -> str | tuple[int, int] | list[int]:
    if value == "all":
        return value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(item) is int for item in value)
        and value[0] >= 1
        and value[1] >= value[0]
    ):
        return value
    raise ValueError(
        "data_range must be 'all' or two increasing, one-based integer indices"
    )


def _expected_draw_counts(
    parameters: dict[str, Any],
) -> tuple[int, int, int]:
    n_chains = parameters["n_MCMC_chains"]
    n_samples = parameters["n_samples_to_save"]
    if n_chains < 1:
        raise ValueError("n_MCMC_chains must be positive")
    if n_samples < 1:
        raise ValueError("n_samples_to_save must be positive")
    if n_samples % n_chains != 0:
        raise ValueError("n_samples_to_save must be divisible by n_MCMC_chains")

    chain_to_plot = parameters["chain_to_plot"]
    if not 1 <= chain_to_plot <= n_chains:
        raise ValueError("chain_to_plot must be between 1 and n_MCMC_chains")

    annealing_constant = parameters["annealing_constant"]
    if annealing_constant <= 0.0:
        raise ValueError("annealing_constant must be positive")

    save_period_float = 12.0 * annealing_constant
    if not save_period_float.is_integer():
        raise ValueError(
            "12 * annealing_constant must be an integer, as required by Julia"
        )
    if parameters["modeling_choice"] == "nonparametric":
        burn_in_float = 11.0 * annealing_constant
        if not burn_in_float.is_integer():
            raise ValueError(
                "11 * annealing_constant must be an integer for the "
                "nonparametric model"
            )
    elif parameters["modeling_choice"] != "parametric":
        raise ValueError(
            "modeling_choice must be either 'nonparametric' or 'parametric'"
        )

    save_period = int(save_period_float)
    samples_per_chain = n_samples // n_chains
    return samples_per_chain * save_period, save_period, n_chains


def _prepare_job(
    job_index: int,
    job: dict[str, Any],
    base_random_seed: int,
) -> dict[str, Any]:
    if "input_file" not in job:
        raise ValueError(f"Trajectory {job_index + 1} is missing 'input_file'")

    input_file = _resolve_path(job["input_file"])
    if not input_file.is_file():
        raise FileNotFoundError(f"Trajectory input does not exist: {input_file}")

    output_directory = _resolve_path(job.get("output_directory", OUTPUT_DIRECTORY))
    overrides = job.get("parameters", {})
    if not isinstance(overrides, dict):
        raise TypeError(
            f"Trajectory {job_index + 1} 'parameters' must be a dictionary"
        )
    unknown_parameters = sorted(set(overrides) - set(DEFAULT_PARAMETERS))
    if unknown_parameters:
        raise ValueError(
            f"Trajectory {job_index + 1} has unknown parameters: "
            + ", ".join(unknown_parameters)
        )

    parameters = dict(DEFAULT_PARAMETERS)
    parameters.update(overrides)
    _validate_parameter_types(parameters)
    total_draws, save_frequency, n_mcmc_chains = _expected_draw_counts(
        parameters
    )

    return {
        "job_index": job_index,
        "input_file": input_file,
        "output_directory": output_directory,
        "total_draws": total_draws,
        "save_frequency": save_frequency,
        "n_mcmc_chains": n_mcmc_chains,
        "configuration": {
            "run": {
                "input_files": [str(input_file)],
                "output_directory": str(output_directory),
                "base_random_seed": base_random_seed,
                "data_range": _validate_data_range(
                    job.get("data_range", "all")
                ),
            },
            "parameters": parameters,
        },
    }


def _prepare_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    next_seed = BASE_RANDOM_SEED
    for job_index, trajectory in enumerate(TRAJECTORIES):
        job = _prepare_job(job_index, trajectory, next_seed)
        jobs.append(job)
        next_seed += job["n_mcmc_chains"] + 1
    return jobs


def _validate_output_paths(jobs: list[dict[str, Any]]) -> None:
    output_paths: dict[Path, Path] = {}
    for job in jobs:
        input_file = job["input_file"]
        output_path = (
            job["output_directory"] / f"mcmc_output_{input_file.stem}.h5"
        )
        if output_path in output_paths:
            other_input = output_paths[output_path]
            raise ValueError(
                "Two trajectories would write the same output file: "
                f"{other_input} and {input_file} -> {output_path}"
            )
        output_paths[output_path] = input_file


def _run_trajectory(
    job: dict[str, Any],
    julia_executable: str,
    progress: _ProgressTracker,
) -> None:
    job_index = job["job_index"]
    job_number = job_index + 1
    input_file = job["input_file"]
    progress.start(job_index)

    try:
        with TemporaryDirectory(
            prefix=f"bnp-walk-multichain-{job_number}-"
        ) as temporary_directory:
            configuration_path = Path(temporary_directory) / "run.toml"
            _write_toml(configuration_path, job["configuration"])

            command = [
                julia_executable,
                "--startup-file=no",
                str(JULIA_SOURCE_DIRECTORY / "main.jl"),
                str(configuration_path),
            ]
            _terminal_print(
                f"Starting trajectory {job_number} with "
                f"{job['n_mcmc_chains']} chains: {input_file}"
            )
            process = subprocess.Popen(
                command,
                cwd=JULIA_SOURCE_DIRECTORY,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                _terminal_print(f"[trajectory {job_number}] {line}", end="")

                draw_match = _DRAW_PATTERN.search(line)
                if draw_match is not None:
                    # Julia prints the draw number immediately before executing
                    # that synchronized draw on all chains.
                    completed_draws = max(int(draw_match.group(1)) - 1, 0)
                    progress.update(job_index, completed_draws)
                    continue

                samples_match = _SAVED_SAMPLES_PATTERN.search(line)
                if samples_match is not None:
                    saved_samples = int(samples_match.group(1))
                    completed_batches = saved_samples // job["n_mcmc_chains"]
                    progress.update(
                        job_index,
                        completed_batches * job["save_frequency"],
                    )

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        progress.finish(job_index)
        _terminal_print(f"Finished trajectory {job_number}: {input_file}")
        progress.report()
    except Exception:
        progress.fail(job_index)
        progress.report()
        raise


def main() -> None:
    julia_executable = os.environ.get("JULIA_EXECUTABLE") or shutil.which("julia")
    if julia_executable is None:
        raise RuntimeError(
            "Julia was not found. Put it on PATH or set JULIA_EXECUTABLE to its path."
        )

    if not TRAJECTORIES:
        raise ValueError("TRAJECTORIES must contain at least one trajectory")
    if MAX_PARALLEL_TRAJECTORIES < 1:
        raise ValueError("MAX_PARALLEL_TRAJECTORIES must be at least 1")
    if PROGRESS_REPORT_INTERVAL_SECONDS <= 0.0:
        raise ValueError("PROGRESS_REPORT_INTERVAL_SECONDS must be positive")
    if type(BASE_RANDOM_SEED) is not int:
        raise TypeError("BASE_RANDOM_SEED must be an integer")

    jobs = _prepare_jobs()
    _validate_output_paths(jobs)

    worker_count = min(MAX_PARALLEL_TRAJECTORIES, len(jobs))
    maximum_internal_workers = sum(
        sorted((job["n_mcmc_chains"] for job in jobs), reverse=True)[
            :worker_count
        ]
    )
    _terminal_print(
        f"Launching {len(jobs)} trajectories with up to {worker_count} "
        f"concurrent Julia instances and {maximum_internal_workers} "
        "internal MCMC workers."
    )

    progress = _ProgressTracker(jobs)
    progress_stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_report_progress_until_stopped,
        args=(progress, progress_stop_event),
        name="progress-reporter",
        daemon=True,
    )
    progress_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_run_trajectory, job, julia_executable, progress)
                for job in jobs
            ]
            # Calling result propagates any Julia failure to this launcher.
            for future in futures:
                future.result()
    finally:
        progress_stop_event.set()
        progress_thread.join()
        progress.report()


if __name__ == "__main__":
    main()
