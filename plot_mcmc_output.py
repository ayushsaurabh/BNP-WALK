#!/usr/bin/env python3
"""Plot BNP-WALK MCMC output using Matplotlib.

The script accepts HDF5 files produced by the Julia sampler and trajectory
files in either of the formats understood by the sampler:

* ``analysis/xy`` (with time step ``analysis/t/xy/dt``), or
* root-level ``x(m)`` and ``y(m)`` (with time step ``dt(s)``).

Example
-------
python plot_mcmc_output.py mcmc_output.h5 trajectory.h5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib

# This script only writes an image, so it must also work on headless workers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MCMC_DATASETS = (
    "mcmc_loads",
    "mcmc_change_points",
    "mcmc_dwell_gamma_shape",
    "mcmc_dwell_gamma_mean",
    "mcmc_position_x",
    "mcmc_position_y",
)


def _dataset_array(h5_file: h5py.File, name: str) -> np.ndarray:
    if name not in h5_file:
        raise KeyError(f"Missing dataset {name!r} in {h5_file.filename}")
    return np.asarray(h5_file[name])


def _sample_major(array: np.ndarray, n_samples: int, name: str) -> np.ndarray:
    """Return a two-dimensional MCMC array with samples on axis zero.

    HDF5.jl reverses matrix dimensions in the on-disk representation.  Thus a
    Julia ``(n_samples, n_parameters)`` matrix normally appears to h5py as
    ``(n_parameters, n_samples)``.  This also accepts sample-major HDF5 files.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name!r} must be two-dimensional; got {array.shape}")
    if array.shape[0] == n_samples:
        return array
    if array.shape[1] == n_samples:
        return array.T
    raise ValueError(
        f"Neither dimension of {name!r} ({array.shape}) matches the "
        f"number of MCMC samples ({n_samples})"
    )


def load_mcmc_output(filename: Path) -> dict[str, np.ndarray]:
    with h5py.File(filename, "r") as h5_file:
        arrays = {name: _dataset_array(h5_file, name) for name in MCMC_DATASETS}

    gamma_shape = np.ravel(arrays["mcmc_dwell_gamma_shape"])
    gamma_mean = np.ravel(arrays["mcmc_dwell_gamma_mean"])
    if gamma_shape.size != gamma_mean.size:
        raise ValueError(
            "mcmc_dwell_gamma_shape and mcmc_dwell_gamma_mean have "
            "different sample counts"
        )
    if gamma_shape.size == 0:
        raise ValueError("The MCMC output contains no saved samples")

    n_samples = gamma_shape.size
    arrays["mcmc_dwell_gamma_shape"] = gamma_shape
    arrays["mcmc_dwell_gamma_mean"] = gamma_mean
    for name in (
        "mcmc_loads",
        "mcmc_change_points",
        "mcmc_position_x",
        "mcmc_position_y",
    ):
        arrays[name] = _sample_major(arrays[name], n_samples, name)

    if arrays["mcmc_loads"].shape != arrays["mcmc_change_points"].shape:
        raise ValueError("mcmc_loads and mcmc_change_points have different shapes")
    if arrays["mcmc_position_x"].shape != arrays["mcmc_position_y"].shape:
        raise ValueError("mcmc_position_x and mcmc_position_y have different shapes")
    return arrays


def _coordinate_from_xy(xy: np.ndarray, coordinate: int) -> np.ndarray:
    """Extract one coordinate from either Julia- or C-oriented analysis/xy."""
    xy = np.asarray(xy)
    if xy.ndim not in (2, 3):
        raise ValueError(f"analysis/xy must be two- or three-dimensional; got {xy.shape}")

    coordinate_axes = [axis for axis, size in enumerate(xy.shape) if size == 2]
    if not coordinate_axes:
        raise ValueError(
            f"Could not identify the x/y dimension (length 2) in analysis/xy {xy.shape}"
        )

    # The established format is (time, xy, object) on disk and (object, xy,
    # time) in Julia.  In both cases the coordinate axis is normally axis 1.
    coordinate_axis = 1 if 1 in coordinate_axes else coordinate_axes[0]
    values = np.take(xy, coordinate, axis=coordinate_axis)
    time_axis = int(np.argmax(values.shape))
    values = np.moveaxis(values, time_axis, 0)
    return np.asarray(values).reshape(values.shape[0], -1)[:, 0]


def _optional_vector(h5_file: h5py.File, name: str) -> np.ndarray | None:
    return np.ravel(np.asarray(h5_file[name], dtype=float)) if name in h5_file else None


def _scalar_dataset(h5_file: h5py.File, names: Sequence[str]) -> float | None:
    for name in names:
        if name in h5_file:
            values = np.asarray(h5_file[name], dtype=float)
            if values.size != 1:
                raise ValueError(f"Expected scalar dataset {name!r}; got {values.shape}")
            return float(values.reshape(-1)[0])
    return None


def load_trajectory(
    filename: Path, start: int | None = None, stop: int | None = None
) -> dict[str, np.ndarray | float | None]:
    with h5py.File(filename, "r") as h5_file:
        if "analysis/xy" in h5_file:
            xy = np.asarray(h5_file["analysis/xy"])
            position_x = _coordinate_from_xy(xy, 0)
            position_y = _coordinate_from_xy(xy, 1)
        elif "x(m)" in h5_file and "y(m)" in h5_file:
            position_x = np.ravel(np.asarray(h5_file["x(m)"], dtype=float))
            position_y = np.ravel(np.asarray(h5_file["y(m)"], dtype=float))
        else:
            raise KeyError(
                f"{filename} must contain either 'analysis/xy' or both "
                "'x(m)' and 'y(m)'"
            )

        dt = _scalar_dataset(h5_file, ("analysis/t/xy/dt", "dt(s)"))
        ground_truth_x = _optional_vector(h5_file, "gt_x(m)")
        ground_truth_y = _optional_vector(h5_file, "gt_y(m)")
        ground_truth_change_points = _optional_vector(
            h5_file, "gt_change_points_time(s)"
        )

    if position_x.size != position_y.size:
        raise ValueError("The trajectory x and y datasets have different lengths")
    selected = slice(start, stop)
    position_x = np.asarray(position_x[selected], dtype=float)
    position_y = np.asarray(position_y[selected], dtype=float)
    if position_x.size < 2:
        raise ValueError("The selected trajectory must contain at least two samples")

    # Preserve missing samples as NaNs so Matplotlib leaves gaps instead of
    # connecting points across missing data.  Center on the first plottable
    # x/y pair; this also handles a missing first sample without contaminating
    # the rest of the trajectory.
    position_x[~np.isfinite(position_x)] = np.nan
    position_y[~np.isfinite(position_y)] = np.nan
    plottable = np.isfinite(position_x) & np.isfinite(position_y)
    if np.any(plottable):
        origin = int(np.flatnonzero(plottable)[0])
        position_x -= position_x[origin]
        position_y -= position_y[origin]
    return {
        "x": position_x,
        "y": position_y,
        "dt": dt,
        "ground_truth_x": ground_truth_x,
        "ground_truth_y": ground_truth_y,
        "ground_truth_change_points": ground_truth_change_points,
    }


def collect_posterior_values(
    mcmc: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    loads = mcmc["mcmc_loads"]
    change_points = mcmc["mcmc_change_points"]
    position_x = mcmc["mcmc_position_x"]
    position_y = mcmc["mcmc_position_y"]

    all_change_points: list[np.ndarray] = []
    all_dwells: list[np.ndarray] = []
    all_position_x: list[np.ndarray] = []
    all_position_y: list[np.ndarray] = []
    all_jump_sizes: list[np.ndarray] = []

    for sample in range(loads.shape[0]):
        active = np.flatnonzero(loads[sample] != 0)
        if active.size < 2:
            continue
        if active[-2] >= position_x.shape[1]:
            raise ValueError(
                "An active dwell index exceeds the width of the position arrays"
            )

        sample_change_points = change_points[sample, active]
        sample_x = position_x[sample, active[:-1]]
        sample_y = position_y[sample, active[:-1]]
        all_change_points.append(sample_change_points)
        all_dwells.append(np.diff(sample_change_points))
        all_position_x.append(sample_x)
        all_position_y.append(sample_y)
        if sample_x.size > 1:
            all_jump_sizes.append(np.hypot(np.diff(sample_x), np.diff(sample_y)))

    if not all_position_x:
        raise ValueError("No samples contain at least two active change points")

    result = {
        "change_points": np.concatenate(all_change_points),
        "dwells": np.concatenate(all_dwells),
        "position_x": np.concatenate(all_position_x),
        "position_y": np.concatenate(all_position_y),
        "jump_sizes": (
            np.concatenate(all_jump_sizes) if all_jump_sizes else np.empty(0)
        ),
    }
    for name, values in result.items():
        result[name] = values[np.isfinite(values)]
    result["dwells"] = result["dwells"][result["dwells"] > 0.0]
    if result["dwells"].size == 0:
        raise ValueError("No positive dwell times were found in the MCMC output")
    return result


def _digamma(value: float) -> float:
    result = 0.0
    while value < 8.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    inverse_squared = inverse * inverse
    return result + math.log(value) - 0.5 * inverse - inverse_squared * (
        1.0 / 12.0
        - inverse_squared
        * (1.0 / 120.0 - inverse_squared * (1.0 / 252.0 - inverse_squared / 240.0))
    )


def _trigamma(value: float) -> float:
    result = 0.0
    while value < 8.0:
        result += 1.0 / (value * value)
        value += 1.0
    inverse = 1.0 / value
    inverse_squared = inverse * inverse
    return result + inverse + 0.5 * inverse_squared + inverse_squared * inverse * (
        1.0 / 6.0
        + inverse_squared
        * (-1.0 / 30.0 + inverse_squared * (1.0 / 42.0 - inverse_squared / 30.0))
    )


def fit_gamma_mle(values: np.ndarray) -> tuple[float, float]:
    """Fit shape and scale with location fixed at zero."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("At least one positive value is required for a Gamma fit")
    mean = float(np.mean(values))
    s = math.log(mean) - float(np.mean(np.log(values)))
    if s <= np.finfo(float).eps:
        # Nearly constant data approach a point mass (shape -> infinity).
        shape = 1.0e8
    else:
        shape = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        for _ in range(50):
            numerator = math.log(shape) - _digamma(shape) - s
            denominator = 1.0 / shape - _trigamma(shape)
            proposed = shape - numerator / denominator
            if not math.isfinite(proposed) or proposed <= 0.0:
                proposed = shape / 2.0
            if abs(proposed - shape) <= 1.0e-12 * max(1.0, shape):
                shape = proposed
                break
            shape = proposed
    return shape, mean / shape


def gamma_pdf(grid: np.ndarray, shape: float, scale: float) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    density = np.zeros_like(grid)
    positive = grid > 0.0
    log_density = (
        (shape - 1.0) * np.log(grid[positive])
        - grid[positive] / scale
        - math.lgamma(shape)
        - shape * math.log(scale)
    )
    density[positive] = np.exp(np.clip(log_density, -745.0, 709.0))
    if shape == 1.0:
        density[~positive] = 1.0 / scale
    return density


def kde2d(
    x: np.ndarray,
    y: np.ndarray,
    points: int = 512,
    bandwidth: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned Gaussian KDE; coordinates and bandwidth use the same units."""
    if points < 32:
        raise ValueError("kde-points must be at least 32")
    if bandwidth <= 0.0:
        raise ValueError("kde-bandwidth must be positive")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        raise ValueError("No finite posterior positions are available for the KDE")

    padding = 4.0 * bandwidth
    x_edges = np.linspace(float(np.min(x)) - padding, float(np.max(x)) + padding, points + 1)
    y_edges = np.linspace(float(np.min(y)) - padding, float(np.max(y)) + padding, points + 1)
    histogram, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges))
    dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]

    # Zero padding prevents the FFT's periodic boundary from wrapping density
    # from one side of the image to the other.
    padded = np.zeros((2 * points, 2 * points), dtype=float)
    padded[:points, :points] = histogram
    frequency_x = np.fft.fftfreq(padded.shape[1], d=dx)
    frequency_y = np.fft.fftfreq(padded.shape[0], d=dy)
    gaussian = np.exp(
        -2.0
        * math.pi**2
        * bandwidth**2
        * (frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2)
    )
    density = np.fft.ifft2(np.fft.fft2(padded) * gaussian).real[:points, :points]
    density = np.maximum(density, 0.0)
    normalization = float(np.sum(density) * dx * dy)
    if normalization > 0.0:
        density /= normalization
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, density


def make_plot(
    mcmc: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray | float | None],
    output: Path,
    *,
    dt: float,
    trajectory_stride: int = 20,
    kde_points: int = 512,
    kde_bandwidth: float = 2.0,
    heatmap_vmax: float | None = 4.5e-4,
    dpi: int = 150,
) -> None:
    posterior = collect_posterior_values(mcmc)
    positions_x_nm = posterior["position_x"] * 1.0e9
    positions_y_nm = posterior["position_y"] * 1.0e9
    kde_x, kde_y, density = kde2d(
        positions_x_nm, positions_y_nm, kde_points, kde_bandwidth
    )

    gamma_shape, gamma_scale = fit_gamma_mle(posterior["dwells"])
    duration = (len(trajectory["x"]) - 1) * dt  # type: ignore[arg-type]
    dwell_limit = 0.05
    dwell_grid = np.linspace(0.0, max(duration, dwell_limit), 1001)
    fitted_density = gamma_pdf(dwell_grid, gamma_shape, gamma_scale)

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    figure = plt.figure(figsize=(24, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 5, height_ratios=(2.4, 1.0))

    axis = figure.add_subplot(grid[0, :])
    extent = (kde_x[0], kde_x[-1], kde_y[0], kde_y[-1])
    image_options: dict[str, object] = {
        "origin": "lower",
        "extent": extent,
        "aspect": "equal",
        "cmap": "viridis",
        "interpolation": "nearest",
    }
    if heatmap_vmax is not None:
        image_options.update(vmin=0.0, vmax=heatmap_vmax)
    axis.imshow(density, **image_options)
    trajectory_x = np.asarray(trajectory["x"]) * 1.0e9
    trajectory_y = np.asarray(trajectory["y"]) * 1.0e9
    sampled_indices = np.arange(0, trajectory_x.size, trajectory_stride)
    missing_indices = np.flatnonzero(
        ~np.isfinite(trajectory_x) | ~np.isfinite(trajectory_y)
    )
    # Retain missing indices even when downsampling so each NaN still breaks
    # the Matplotlib line rather than allowing a segment to bridge the gap.
    plot_indices = np.union1d(sampled_indices, missing_indices)
    axis.plot(
        trajectory_x[plot_indices],
        trajectory_y[plot_indices],
        color="white",
        linewidth=0.5,
        alpha=0.8,
        label="Data",
    )
    ground_truth_x = trajectory["ground_truth_x"]
    ground_truth_y = trajectory["ground_truth_y"]
    if ground_truth_x is not None and ground_truth_y is not None:
        axis.scatter(
            np.asarray(ground_truth_x) * 1.0e9,
            np.asarray(ground_truth_y) * 1.0e9,
            color="red",
            s=25,
            label="Ground truth dwell positions",
            zorder=3,
        )
    axis.set(xlabel="x (nm)", ylabel="y (nm)")
    axis.set_xlim(kde_x[0], kde_x[-1])
    axis.set_ylim(kde_y[0], kde_y[-1])
    axis.set_title(
        f"Bayesian inference of trajectory using {mcmc['mcmc_loads'].shape[0]} MCMC samples",
        fontsize=20,
    )
    legend = axis.legend(loc="upper left", frameon=False)
    for text in legend.get_texts():
        text.set_color("white")

    axis = figure.add_subplot(grid[1, 0])
    axis.hist(posterior["change_points"], bins=100)
    ground_truth_change_points = trajectory["ground_truth_change_points"]
    if ground_truth_change_points is not None:
        axis.vlines(
            np.asarray(ground_truth_change_points),
            *axis.get_ylim(),
            color="red",
            linewidth=0.8,
            label="Ground truth",
        )
        axis.legend(loc="upper right", frameon=False)
    axis.set(xlabel="Change point (s)", ylabel="Counts", xlim=(0.0, duration))

    axis = figure.add_subplot(grid[1, 1])
    axis.hist(posterior["dwells"], bins=200, density=True)
    axis.plot(
        dwell_grid,
        fitted_density,
        color="red",
        linewidth=2.0,
        label=(
            rf"Gamma fit: $\alpha={gamma_shape:.2f}$, "
            rf"mean $={gamma_shape * gamma_scale:.4g}$ s"
        ),
    )
    axis.set(xlabel="Dwell time (s)", ylabel="PDF", xlim=(0.0, dwell_limit))
    axis.legend(loc="upper right", frameon=False, fontsize=9)

    axis = figure.add_subplot(grid[1, 2])
    if posterior["jump_sizes"].size:
        axis.hist(posterior["jump_sizes"] * 1.0e9, bins=200, density=True)
    axis.set(xlabel="Jump size (nm)", ylabel="PDF", xlim=(0.0, 50.0))

    axis = figure.add_subplot(grid[1, 3])
    sampled_gamma_shape = mcmc["mcmc_dwell_gamma_shape"]
    sampled_gamma_shape = sampled_gamma_shape[np.isfinite(sampled_gamma_shape)]
    axis.hist(sampled_gamma_shape, bins=10, density=True)
    axis.set(xlabel="Dwell-time Gamma shape", ylabel="PDF", xlim=(0.0, 5.0))

    axis = figure.add_subplot(grid[1, 4])
    sampled_gamma_mean = mcmc["mcmc_dwell_gamma_mean"]
    sampled_gamma_mean = sampled_gamma_mean[np.isfinite(sampled_gamma_mean)]
    axis.hist(sampled_gamma_mean, bins=40, density=True)
    axis.set(xlabel="Dwell-time Gamma mean (s)", ylabel="PDF", xlim=(0.0, dwell_limit))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot BNP-WALK MCMC output and trajectory data with Matplotlib."
    )
    parser.add_argument("mcmc_output", type=Path, help="MCMC output HDF5 file")
    parser.add_argument("trajectory", type=Path, help="Trajectory HDF5 file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image (default: analysis_plot_<trajectory stem>.png)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Sampling interval in seconds (overrides the trajectory HDF5 value)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First trajectory sample, using zero-based Python indexing",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Exclusive final trajectory sample, using zero-based Python indexing",
    )
    parser.add_argument("--trajectory-stride", type=int, default=20)
    parser.add_argument("--kde-points", type=int, default=512)
    parser.add_argument("--kde-bandwidth", type=float, default=2.0, help="KDE bandwidth in nm")
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=4.5e-4,
        help="Maximum KDE color value; pass 0 for automatic scaling",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mcmc_output.is_file():
        raise FileNotFoundError(f"MCMC output file not found: {args.mcmc_output}")
    if not args.trajectory.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")
    if args.trajectory_stride < 1:
        raise ValueError("trajectory-stride must be at least 1")
    if args.dpi < 1:
        raise ValueError("dpi must be positive")

    mcmc = load_mcmc_output(args.mcmc_output)
    trajectory = load_trajectory(args.trajectory, args.start, args.stop)
    dt = args.dt if args.dt is not None else trajectory["dt"]
    if dt is None:
        raise ValueError(
            "No sampling interval was found at 'analysis/t/xy/dt' or 'dt(s)'; "
            "provide one with --dt"
        )
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite, positive number")

    output = args.output
    if output is None:
        output = args.mcmc_output.parent / f"analysis_plot_{args.trajectory.stem}.png"
    heatmap_vmax = None if args.heatmap_vmax == 0.0 else args.heatmap_vmax
    if heatmap_vmax is not None and heatmap_vmax < 0.0:
        raise ValueError("heatmap-vmax must be nonnegative")
    make_plot(
        mcmc,
        trajectory,
        output,
        dt=dt,
        trajectory_stride=args.trajectory_stride,
        kde_points=args.kde_points,
        kde_bandwidth=args.kde_bandwidth,
        heatmap_vmax=heatmap_vmax,
        dpi=args.dpi,
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
