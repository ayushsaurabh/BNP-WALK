"""Utilities for reading trajectories from BNP-WALK MCMC output."""

from __future__ import annotations

from os import PathLike

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray


def read_xy_trajectories(
    file_name: str | PathLike[str], t: ArrayLike
) -> NDArray[np.float64]:
    """Return the sampled latent trajectories at the requested times.

    Parameters
    ----------
    file_name
        HDF5 output file written by ``main.jl``.
    t
        One-dimensional sequence of times in seconds. Times may be in any
        order. A time outside a sample's modeled interval is clamped to that
        sample's nearest endpoint.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, M, 2, 1)``, where ``N`` is the number of saved
        MCMC samples and ``M = len(t)``. The third axis contains x and y,
        respectively, in the position units stored in the HDF5 file (meters
        for the supplied model).

    Notes
    -----
    The inferred trajectory is piecewise constant. At a change-point time the
    position of the new dwell is returned. At the final boundary, the position
    of the final dwell is returned.

    Julia's HDF5 writer exposes the two-dimensional datasets to h5py with
    reversed axes: candidate change points/dwells are rows and MCMC samples
    are columns.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("t must be a one-dimensional sequence")
    if not np.all(np.isfinite(times)):
        raise ValueError("t must contain only finite values")

    with h5py.File(file_name, "r") as output_file:
        loads = np.asarray(output_file["mcmc_loads"])
        change_points = np.asarray(output_file["mcmc_change_points"])
        position_x = np.asarray(output_file["mcmc_position_x"])
        position_y = np.asarray(output_file["mcmc_position_y"])

    if loads.ndim != 2 or change_points.shape != loads.shape:
        raise ValueError(
            "mcmc_loads and mcmc_change_points must be equally shaped 2-D arrays"
        )
    if position_x.ndim != 2 or position_y.shape != position_x.shape:
        raise ValueError(
            "mcmc_position_x and mcmc_position_y must be equally shaped 2-D arrays"
        )
    if position_x.shape != (loads.shape[0] - 1, loads.shape[1]):
        raise ValueError("the HDF5 datasets do not have the layout written by main.jl")

    n_samples = loads.shape[1]
    trajectories = np.empty((n_samples, times.size, 2, 1), dtype=np.float64)

    for sample in range(n_samples):
        active = np.flatnonzero(loads[:, sample])
        if active.size < 2:
            raise ValueError(f"MCMC sample {sample} has fewer than two active boundaries")

        boundaries = change_points[active, sample]
        if np.any(np.diff(boundaries) <= 0.0):
            raise ValueError(
                f"active change-point times are not strictly increasing in sample {sample}"
            )
        # searchsorted(..., side="right") assigns a boundary to the dwell that
        # starts there. Times outside this sample's range use its nearest
        # endpoint, and clipping the dwell index handles the final boundary.
        sample_times = np.clip(times, boundaries[0], boundaries[-1])
        dwell = np.searchsorted(boundaries[1:], sample_times, side="right")
        dwell = np.minimum(dwell, active.size - 2)
        position_index = active[:-1][dwell]

        trajectories[sample, :, 0, 0] = position_x[position_index, sample]
        trajectories[sample, :, 1, 0] = position_y[position_index, sample]

    return trajectories
