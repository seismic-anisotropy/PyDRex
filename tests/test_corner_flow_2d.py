"""PyDRex: 2D corner flow tests.

NOTE: In scipy, rotations are represented as a matrix
that transforms [1, 0, 0] to the new a-axis vector.
We need the inverse of that rotation,
which represents the change of coordinates
from the grain-local to the global (Eulerian) frame.

"""
import numpy as np
# from numpy import random as rn
# from scipy import linalg as la
from pydrex import deformation_mechanism as _defmech
from pydrex import diagnostics as _diagnostics
from pydrex import minerals as _minerals
from pydrex import visualisation as _vis
from scipy.spatial.transform import Rotation


class TestOlivineA:
    """Tests for pure A-type olivine polycrystals in 2D corner flows."""

    def test_cornerXZ_init_random(
        self,
        params_Kaminski2004_fig4_squares,
        outdir,
    ):
        r"""Test CPO evolution along 4 corner flow pathlines from -Z to +X.

        Initial condition: fully random orientations in all 4 `Mineral`s.
        The flow field is defined by:
        $$
        u = \frac{2 U r}{π}(θ\sinθ - \cosθ) ⋅ \hat{r} + \frac{2 U}{π}θ\cosθ ⋅ \hat{θ}
        $$
        Note that there is a missing factor of $$r$$ in [Kaminski 2002].

        where $r = θ = 0$ points vertically downwards along the ridge axis
        and $θ = π/2$ points along the surface. $$U$$ is the half spreading velocity.
        Streamlines for the flow obey:
        $$
        \psi = \frac{2 U r}{π}θ\cosθ
        $$

        The velocity gradient in the Cartesian (x, z) basis is given by:
        $$
        L = \frac{2 U θ}{π} ⋅
        \begin{matrix}
            \sin^{3}θ - \sin^{2}θ\cosθ & 0 & -\sin^{2}θ\cosθ + \sinθ\cos^{2}θ \\
            0 & 0 & 0 \\
            -\sin^{2}θ\cosθ + \sinθ\cos^{2}θ & 0 & \sinθ\cos^{2}θ - \cos^{3}θ
        \end{matrix}
        $$


        Similar to Fig. 5 in [Kaminski 2002].

        [Kaminski 2002](https://doi.org/10.1029/2001GC000222)

        """
        # Plate velocity (half spreading rate), convert cm/yr to m/s.
        plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
        timestep = 0.2 / plate_velocity
        domain_height = 1.0  # Normalised to olivine-spinel transition.
        domain_width = 5.0

        n_grains = 1000
        orientations_init = Rotation.random(1000).as_matrix()

        # Optional plotting setup.
        if outdir is not None:
            labels = []
            angles = []
            indices = []
            r_vals = []
            θ_vals = []

        # Note: θ values are in radians.
        for distance_init in (0.25, 0.5, 1.0, 2.0):
            mineral = _minerals.Mineral(
                _minerals.MineralPhase.olivine,
                _minerals.OlivineFabric.A,
                _defmech.Regime.dislocation,
                n_grains=n_grains,
                fractions_init=np.full(n_grains, 1 / n_grains),
                orientations_init=orientations_init,
            )
            r_init = np.sqrt(domain_height**2 + distance_init**2)
            θ_init = np.arcsin(distance_init / r_init)
            r_current = r_init
            θ_current = θ_init
            r_all = []
            θ_all = []
            deformation_gradient = np.eye(3)  # Undeformed initial state.
            # While the polycrystal is inside the domain.
            while r_current * np.sin(θ_current) < domain_width:
                prefactor = 2 * plate_velocity / np.pi
                sinθ = np.sin(θ_current)
                cosθ = np.cos(θ_current)
                velocity_gradient = prefactor * θ_current * np.array([
                    [sinθ ** 3 - sinθ ** 2 * cosθ, 0, -sinθ ** 2 * cosθ + sinθ * cosθ ** 2],
                    [0, 0, 0],
                    [-sinθ ** 2 * cosθ + sinθ * cosθ ** 2, 0, sinθ * cosθ ** 2 - cosθ ** 3],
                ])

                # timescale = 1 / (
                #     la.eigvalsh(velocity_gradient + velocity_gradient.transpose()).max()
                #     / 2
                # )

                deformation_gradient = mineral.update_orientations(
                    params_Kaminski2004_fig4_squares,
                    deformation_gradient,
                    velocity_gradient,
                    integration_time=timestep
                )

                if outdir is not None:
                    r_all.append(r_current)
                    θ_all.append(θ_current)

                θ_current += (
                    prefactor * θ_current * np.cos(θ_current) * timestep
                )
                r_current += (
                    prefactor * r_current
                    * (θ_current * np.sin(θ_current) - np.cos(θ_current))
                    * timestep
                )

            n_timesteps = len(mineral._orientations)
            misorient_angles = np.zeros(n_timesteps)
            misorient_indices = np.zeros(n_timesteps)
            # Loop over first dimension (time steps) of orientations.
            for idx, matrices in enumerate(mineral._orientations):
                orientations_resampled, _ = _diagnostics.resample_orientations(
                    matrices, mineral._fractions[idx]
                )
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.get_primary_axis(mineral.fabric),
                )
                misorient_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean, [1, 0, 0],
                )
                misorient_indices[idx] = _diagnostics.misorientation_index(
                    orientations_resampled
                )

            # # Check for mostly smoothly decreasing misalignment.
            # angles_diff = np.diff(misorient_angles)
            # assert np.max(angles_diff) < 3.2
            # assert np.min(angles_diff) > -7.5
            # assert np.sum(angles_diff) < -25.0

            if outdir is not None:
                mineral.save(f"{outdir}/cornerXZ_olivineA_{str(distance_init).replace('.', 'd')}.npz")
                labels.append(fr"$x_{0}$ = {distance_init}")
                angles.append(misorient_angles)
                indices.append(misorient_indices)
                r_vals.append(r_all)
                θ_vals.append(θ_all)

        if outdir is not None:
            _vis.corner_flow_2d(
                angles,
                indices,
                r_vals,
                θ_vals,
                timestep=timestep,
                savefile=f"{outdir}/cornerXZ_single_olivineA.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
            )
