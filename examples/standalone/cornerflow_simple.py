#!/usr/bin/env python
import doctest
import functools as ft
import logging
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np

import pydrex
from pydrex import pathlines
from pydrex import visualisation as vis


# Define simulation parameters. Check to ensure serializability with a simple doctest.
@dataclass(frozen=True)
class Parameters(pydrex.DefaultParams):  # Inherit default PyDRex parameters.
    """Parameters for a simple cornerflow example.

    >>> params = Parameters()
    >>> # Evaluating hash() will raise an error if there are mutable attributes.
    >>> isinstance(hash(params), int)
    True

    """

    # Add parameters that are specific to this simulation.
    # You MUST use type annotations, otherwise these will be set as class attributes!
    # <https://github.com/python/mypy/issues/17666>
    plate_speed: float = 2.0 / (100 * 365 * 86400)
    domain_height: float = 2e5
    domain_width: float = 1e6
    n_timesteps: int = 50
    pathline_ends: tuple = (-0.1, -0.3, -0.54, -0.78)  # % of domain height, negative z.
    domain_coords: tuple = ("X", "Z")
    max_strain: float = 10.0
    out_figure: str = "cornerflow2d_simple_example.png"
    out_logfile: str = "cornerflow2d_simple_example.log"


doctest.testmod()  # Run doctests for serializability check.


# Initialise parameter store. Optionally modify default PyDRex parameters.
params = Parameters(
    phase_assemblage=(pydrex.MineralPhase.olivine, pydrex.MineralPhase.enstatite),
    phase_fractions=(0.7, 0.3),
    gbm_mobility=10,
    number_of_grains=5000,
)

# Get velocity and velocity gradient callables.
f_velocity, f_velocity_grad = pydrex.velocity.corner_2d(
    *params.domain_coords, params.plate_speed
)

# Turn params.pathline_ends into an array of 3D coordinates.
final_locations = [
    np.array([params.domain_width, 0.0, z * params.domain_height])
    for z in params.pathline_ends
]


# Set up CPO computation for a pathline.
def run(params, f_velocity, f_velocity_grad, min_coords, max_coords, final_location):
    # Get pydrex logger, prefer logging messages instead of using `print()`.
    logger = logging.getLogger("pydrex")
    # Create mineral phase. Each phase is tracked separately.
    olA = pydrex.Mineral(
        pydrex.MineralPhase.olivine,
        pydrex.MineralFabric.olivine_A,
        pydrex.DeformationRegime.matrix_dislocation,
        n_grains=params.number_of_grains,
    )
    ens = pydrex.Mineral(
        pydrex.MineralPhase.enstatite,
        pydrex.MineralFabric.enstatite_AB,
        pydrex.DeformationRegime.matrix_dislocation,
        n_grains=params.number_of_grains,
    )
    # Calculate pathline by solving backwards from the final location.
    timestamps, f_position = pathlines.get_pathline(
        final_location,
        f_velocity,
        f_velocity_grad,
        min_coords,
        max_coords,
        max_strain=params.max_strain,
        regular_steps=params.n_timesteps,
    )
    positions = [f_position(t) for t in timestamps]
    velocity_gradients = [f_velocity_grad(np.nan, np.asarray(x)) for x in positions]

    # All minerals start undeformed, at zero strain.
    deformation_gradient = np.identity(3)
    strains = np.empty_like(timestamps)
    strains[0] = 0

    # Main CPO computation loop.
    for i, time in enumerate(timestamps[:-1], start=1):
        strains[i] = strains[i - 1] + pydrex.utils.strain_increment(
            timestamps[i] - time, velocity_gradients[i]
        )
        logger.info(
            "path_id = %d; step = %d/%d (ε = %.2f)",
            hash(abs(final_location[-1])),
            i,
            params.n_timesteps,
            strains[i],
        )
        deformation_gradient = pydrex.update_all(
            (olA, ens),
            params.as_dict(),
            deformation_gradient,
            f_velocity_grad,
            pathline=(time, timestamps[i], f_position),
        )
    return timestamps, positions, strains, (olA, ens), deformation_gradient


# Set up storage for results.
results: list[dict] = []
all_strains = []
all_positions = []

# Get a multiprocessing Pool and record which backend is being used.
Pool, HAS_RAY = pydrex.utils.import_proc_pool()

# Solve CPO on multiple pathlines in parallel using multiprocessing.
with pydrex.io.logfile_enable(params.out_logfile):
    # Using more than 4 cores is not needed and will introduce wasted overhead.
    n_cpus = min(4, pydrex.utils.default_ncpus())
    with Pool(processes=n_cpus) as pool:
        _run = ft.partial(
            run,
            params,
            f_velocity,
            f_velocity_grad,
            np.array([0.0, 0.0, -params.domain_height]),
            np.array([params.domain_width, 0.0, 0.0]),
            # Different final_location depending on pathline.
        )
        for i, out in enumerate(pool.imap(_run, final_locations)):
            times, positions, strains, (olA, ens), deformation_gradient = out
            # Only 1 phase here, 100% olivine.
            avg_stiffnesses = pydrex.minerals.voigt_averages(
                (olA, ens), (olA.phase, ens.phase), params.phase_fractions
            )
            result = pydrex.diagnostics.elasticity_components(avg_stiffnesses)
            all_strains.append(strains)
            all_positions.append(positions)
            results.append(result)

# Simple visualisation using provided plotting function.
fig = vis.figure(figsize=(10, 3))
ax = fig.add_subplot()
markers = ("s", "o", "v", "*")
cmaps = ["cmc.batlow_r"] * len(markers)

for i, z_exit in enumerate(np.array(params.pathline_ends) * params.domain_height):
    velocity_grid_res = None  # Resolution of grid used to compute streamlines.
    density = None  # Density of actual plotted streamlines.
    if i == 0:  # Only plot velocity streamlines once.
        velocity_grid_res = [50, 10]
        density = [0.05 * r for r in velocity_grid_res]

    # Set up plotting function arguments.
    velocity = (f_velocity, velocity_grid_res)
    geometry = (
        all_positions[i],
        [0, -params.domain_height],
        [params.domain_width + 1e5, 0],
    )
    ref_axes = "".join(params.domain_coords)

    # The M-index (`pydrex.diagnostics.misoientation_index`) is a much better measure of
    # texture strength, but is very expensive to compute. For this simple example, we
    # just use the percentage of the average elastic tensor that deviates from the
    # closest isotropic elastic tensor.
    cpo = (results[i]["percent_anisotropy"], results[i]["hexagonal_axis"])

    norm = mpl.colors.Normalize(vmin=all_strains[0][0], vmax=all_strains[0][-1])
    fig, ax, q, s = vis.steady_box2d(
        ax,
        velocity,
        geometry,
        ref_axes,
        cpo,
        colors=all_strains[i],
        marker=markers[i],
        cmap=cmaps[i],
        norm=norm,
        density=density,
    )
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[0]),
    ax=ax,
    aspect=75,
    location="top",
    label="Strain (ε)",
)
fig.savefig(params.out_figure)
