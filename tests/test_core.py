"""> PyDRex: Tests for core D-Rex routines."""

import contextlib as cl

import numpy as np
from numpy import testing as nt
from scipy.spatial.transform import Rotation

from pydrex import core as _core
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import visualisation as _vis
from pydrex import utils as _utils

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "core"


class TestDislocationCreepOPX:
    """Single-grain orthopyroxene crystallographic rotation rate tests."""

    class_id = "dislocation_creep_OPX"

    def test_shear_dudz(self, outdir):
        test_id = "dudz"
        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
        with optional_logging:
            for θ in np.mgrid[0 : 2 * np.pi : 360j]:
                _log.debug("θ (°): %s", np.rad2deg(θ))
                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.enstatite,
                    fabric=_core.MineralFabric.enstatite_AB,
                    n_grains=1,
                    orientations=Rotation.from_rotvec([[0, θ, 0]]).as_matrix(),
                    fractions=np.array([1.0]),
                    strain_rate=np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                    velocity_gradient=np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]]),
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                cosθ = np.cos(θ)
                cos2θ = np.cos(2 * θ)
                sinθ = np.sin(θ)
                target_orientations_diff = (1 + cos2θ) * np.array(
                    [[sinθ, 0, -cosθ], [0, 0, 0], [cosθ, 0, sinθ]]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff
                )
                assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_shear_dvdx(self, outdir):
        test_id = "dvdx"
        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
        with optional_logging:
            for θ in np.linspace(0, 2 * np.pi, 361):
                _log.debug("θ (°): %s", np.rad2deg(θ))
                orientations = Rotation.from_rotvec([[0, 0, θ]]).as_matrix()
                deformation_rate = _core._get_deformation_rate(
                    _core.MineralPhase.enstatite,
                    orientations[0],
                    np.array([0, 0, 0, 0]),
                )
                np.testing.assert_allclose(deformation_rate.flatten(), np.zeros(9))
                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.enstatite,
                    fabric=_core.MineralFabric.enstatite_AB,
                    n_grains=1,
                    orientations=orientations,
                    fractions=np.array([1.0]),
                    strain_rate=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                    velocity_gradient=np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]]),
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                # Can't activate the (100)[001] slip system, no plastic deformation.
                # Only passive (rigid-body) rotation due to the velocity gradient.
                sinθ = np.sin(θ)
                cosθ = np.cos(θ)
                target_orientations_diff = np.array(
                    [[sinθ, cosθ, 0], [-cosθ, sinθ, 0], [0, 0, 0]]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff, atol=1e-15
                )
                assert np.isclose(np.sum(fractions_diff), 0.0)


class TestDislocationCreepOlivineA:
    """Single-grain A-type olivine analytical rotation rate tests."""

    class_id = "dislocation_creep_OlA"

    def test_shear_dvdx_slip_010_100(self, outdir):
        r"""Single grain of A-type olivine, slip on (010)[100].

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dvdx_010_100"
        nondim_velocity_gradient = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]])
        # Strain rate is 0.5*(L + Lᵀ).
        nondim_strain_rate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
            initial_angles = []
            rotation_rates = []
            target_rotation_rates = []

        with optional_logging:
            for θ in np.mgrid[0 : 2 * np.pi : 3600j]:
                _log.debug("θ (°): %s", np.rad2deg(θ))
                # Initial grain rotations around Z (anti-clockwise).
                initial_orientations = Rotation.from_rotvec([[0, 0, θ]])
                orientation_matrix = initial_orientations[0].as_matrix()
                slip_invariants = _core._get_slip_invariants(
                    nondim_strain_rate, orientation_matrix
                )
                _log.debug("slip invariants: %s", slip_invariants)

                crss = _core.get_crss(
                    _core.MineralPhase.olivine,
                    _core.MineralFabric.olivine_A,
                )
                slip_indices = np.argsort(np.abs(slip_invariants / crss))
                slip_system = _minerals.OLIVINE_SLIP_SYSTEMS[slip_indices[-1]]
                assert slip_system == ([0, 1, 0], [1, 0, 0])

                slip_rates = _core._get_slip_rates_olivine(
                    slip_invariants, slip_indices, crss, 3.5
                )
                _log.debug("slip rates: %s", slip_rates)

                deformation_rate = _core._get_deformation_rate(
                    _core.MineralPhase.olivine,
                    orientation_matrix,
                    slip_rates,
                )
                _log.debug("deformation rate:\n%s", deformation_rate)

                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.olivine,
                    fabric=_core.MineralFabric.olivine_A,
                    n_grains=1,
                    orientations=initial_orientations.as_matrix(),
                    fractions=np.array([1.0]),
                    strain_rate=nondim_strain_rate,
                    velocity_gradient=nondim_velocity_gradient,
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                cosθ = np.cos(θ)
                cos2θ = np.cos(2 * θ)
                sinθ = np.sin(θ)
                target_orientations_diff = np.array(
                    [
                        [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
                        [-cosθ * (1 + cos2θ), sinθ * (1 + cos2θ), 0],
                        [0, 0, 0],
                    ]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff
                )
                if outdir is not None:
                    initial_angles.append(np.rad2deg(θ))
                    rotation_rates.append(
                        np.sqrt(
                            orientations_diff[0][0, 0] ** 2
                            + orientations_diff[0][0, 1] ** 2
                        )
                    )
                    target_rotation_rates.append(1 + cos2θ)
                assert np.isclose(np.sum(fractions_diff), 0.0)

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 2 / 3))
            axs = fig.subplot_mosaic(  # Put the legend on a special axes.
                [["legend"], ["spin"]], gridspec_kw={"height_ratios": [0.1, 1]}
            )
            ax = axs["spin"]
            fig, ax, colors = _vis.spin(
                ax,
                initial_angles[::25],
                rotation_rates[::25],
                initial_angles,
                target_rotation_rates,
                shear_axis=90,
            )
            _utils.redraw_legend(
                ax,
                legendax=axs["legend"],
                loc="upper center",
                ncols=3,
                mode="expand",
            )
            fig.savefig(
                _io.resolve_path(f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.pdf")
            )

    def test_shear_dudz_slip_001_100(self, outdir):
        r"""Single grain of A-type olivine, slip on (001)[100].

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dudz_001_100"
        nondim_velocity_gradient = np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]])
        # Strain rate is 0.5*(L + Lᵀ).
        nondim_strain_rate = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
            initial_angles = []
            rotation_rates = []
            target_rotation_rates = []

        with optional_logging:
            for θ in np.mgrid[0 : 2 * np.pi : 360j]:
                _log.debug("θ (°): %s", np.rad2deg(θ))
                # Initial grain rotations around Y (anti-clockwise).
                initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
                orientation_matrix = initial_orientations[0].as_matrix()
                slip_invariants = _core._get_slip_invariants(
                    nondim_strain_rate, orientation_matrix
                )
                _log.debug("slip invariants: %s", slip_invariants)

                crss = _core.get_crss(
                    _core.MineralPhase.olivine,
                    _core.MineralFabric.olivine_A,
                )
                slip_indices = np.argsort(np.abs(slip_invariants / crss))
                slip_system = _minerals.OLIVINE_SLIP_SYSTEMS[slip_indices[-1]]
                assert slip_system == ([0, 0, 1], [1, 0, 0])

                slip_rates = _core._get_slip_rates_olivine(
                    slip_invariants, slip_indices, crss, 3.5
                )
                _log.debug("slip rates: %s", slip_rates)

                deformation_rate = _core._get_deformation_rate(
                    _core.MineralPhase.olivine,
                    orientation_matrix,
                    slip_rates,
                )
                _log.debug("deformation rate:\n%s", deformation_rate)

                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.olivine,
                    fabric=_core.MineralFabric.olivine_A,
                    n_grains=1,
                    orientations=initial_orientations.as_matrix(),
                    fractions=np.array([1.0]),
                    strain_rate=nondim_strain_rate,
                    velocity_gradient=nondim_velocity_gradient,
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                cosθ = np.cos(θ)
                cos2θ = np.cos(2 * θ)
                sinθ = np.sin(θ)
                target_orientations_diff = np.array(
                    [
                        [-sinθ * (cos2θ - 1), 0, cosθ * (cos2θ - 1)],
                        [0, 0, 0],
                        [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
                    ]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff
                )
                if outdir is not None:
                    initial_angles.append(np.rad2deg(θ))
                    rotation_rates.append(
                        np.sqrt(
                            orientations_diff[0][0, 0] ** 2
                            + orientations_diff[0][0, 2] ** 2
                        )
                    )
                    target_rotation_rates.append(1 - cos2θ)
                assert np.isclose(np.sum(fractions_diff), 0.0)

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 2 / 3))
            axs = fig.subplot_mosaic(  # Put the legend on a special axes.
                [["legend"], ["spin"]], gridspec_kw={"height_ratios": [0.1, 1]}
            )
            ax = axs["spin"]
            fig, ax, colors = _vis.spin(
                ax,
                initial_angles[::5],
                rotation_rates[::5],
                initial_angles,
                target_rotation_rates,
                shear_axis=0,
            )
            _utils.redraw_legend(
                ax,
                legendax=axs["legend"],
                loc="upper center",
                ncols=3,
                mode="expand",
            )
            fig.savefig(
                _io.resolve_path(f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.pdf")
            )

    def test_shear_dwdx_slip_001_100(self, outdir):
        r"""Single grain of A-type olivine, slip on (001)[100].

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 0 & 0 & 0 \cr 2 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dwdx_001_100"
        nondim_velocity_gradient = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0]])
        # Strain rate is 0.5*(L + Lᵀ).
        nondim_strain_rate = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
            initial_angles = []
            rotation_rates = []
            target_rotation_rates = []

        with optional_logging:
            for θ in np.mgrid[0 : 2 * np.pi : 360j]:
                _log.debug("θ (°): %s", np.rad2deg(θ))
                # Initial grain rotations around Y (anti-clockwise).
                initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
                orientation_matrix = initial_orientations[0].as_matrix()
                slip_invariants = _core._get_slip_invariants(
                    nondim_strain_rate, orientation_matrix
                )
                _log.debug("slip invariants: %s", slip_invariants)

                crss = _core.get_crss(
                    _core.MineralPhase.olivine,
                    _core.MineralFabric.olivine_A,
                )
                slip_indices = np.argsort(np.abs(slip_invariants / crss))
                slip_system = _minerals.OLIVINE_SLIP_SYSTEMS[slip_indices[-1]]
                assert slip_system == ([0, 0, 1], [1, 0, 0])

                slip_rates = _core._get_slip_rates_olivine(
                    slip_invariants, slip_indices, crss, 3.5
                )
                _log.debug("slip rates: %s", slip_rates)

                deformation_rate = _core._get_deformation_rate(
                    _core.MineralPhase.olivine,
                    orientation_matrix,
                    slip_rates,
                )
                _log.debug("deformation rate:\n%s", deformation_rate)

                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.olivine,
                    fabric=_core.MineralFabric.olivine_A,
                    n_grains=1,
                    orientations=initial_orientations.as_matrix(),
                    fractions=np.array([1.0]),
                    strain_rate=nondim_strain_rate,
                    velocity_gradient=nondim_velocity_gradient,
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                cosθ = np.cos(θ)
                cos2θ = np.cos(2 * θ)
                sinθ = np.sin(θ)
                target_orientations_diff = np.array(
                    [
                        [-sinθ * (1 + cos2θ), 0, cosθ * (1 + cos2θ)],
                        [0, 0, 0],
                        [-cosθ * (1 + cos2θ), 0, -sinθ * (1 + cos2θ)],
                    ]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff
                )
                if outdir is not None:
                    initial_angles.append(np.rad2deg(θ))
                    rotation_rates.append(
                        np.sqrt(
                            orientations_diff[0][0, 0] ** 2
                            + orientations_diff[0][0, 2] ** 2
                        )
                    )
                    target_rotation_rates.append(1 + cos2θ)
                assert np.isclose(np.sum(fractions_diff), 0.0)

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 2 / 3))
            axs = fig.subplot_mosaic(  # Put the legend on a special axes.
                [["legend"], ["spin"]], gridspec_kw={"height_ratios": [0.1, 1]}
            )
            ax = axs["spin"]
            fig, ax, colors = _vis.spin(
                ax,
                initial_angles[::5],
                rotation_rates[::5],
                initial_angles,
                target_rotation_rates,
                shear_axis=90,
            )
            _utils.redraw_legend(
                ax,
                legendax=axs["legend"],
                loc="upper center",
                ncols=3,
                mode="expand",
            )
            fig.savefig(
                _io.resolve_path(f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.pdf")
            )

    def test_shear_dvdz_slip_010_001(self, outdir):
        r"""Single grain of A-type olivine, slip on (010)[001].

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 0 & 0 & 2 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dvdz_010_001"
        nondim_velocity_gradient = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 0]])
        # Strain rate is 0.5*(L + Lᵀ).
        nondim_strain_rate = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.log"
            )
            initial_angles = []
            rotation_rates = []
            target_rotation_rates = []

        with optional_logging:
            for θ in np.mgrid[0 : 2 * np.pi : 360j]:
                _log.debug("θ (°): %s", np.rad2deg(θ))
                # Initial grain rotations around X (anti-clockwise).
                initial_orientations = Rotation.from_rotvec([[θ, 0, 0]])
                orientation_matrix = initial_orientations[0].as_matrix()
                slip_invariants = _core._get_slip_invariants(
                    nondim_strain_rate, orientation_matrix
                )
                _log.debug("slip invariants: %s", slip_invariants)

                crss = _core.get_crss(
                    _core.MineralPhase.olivine,
                    _core.MineralFabric.olivine_A,
                )
                slip_indices = np.argsort(np.abs(slip_invariants / crss))
                slip_system = _minerals.OLIVINE_SLIP_SYSTEMS[slip_indices[-1]]
                assert slip_system == ([0, 1, 0], [0, 0, 1])

                slip_rates = _core._get_slip_rates_olivine(
                    slip_invariants, slip_indices, crss, 3.5
                )
                _log.debug("slip rates: %s", slip_rates)

                deformation_rate = _core._get_deformation_rate(
                    _core.MineralPhase.olivine,
                    orientation_matrix,
                    slip_rates,
                )
                _log.debug("deformation rate:\n%s", deformation_rate)

                orientations_diff, fractions_diff = _core.derivatives(
                    phase=_core.MineralPhase.olivine,
                    fabric=_core.MineralFabric.olivine_A,
                    n_grains=1,
                    orientations=initial_orientations.as_matrix(),
                    fractions=np.array([1.0]),
                    strain_rate=nondim_strain_rate,
                    velocity_gradient=nondim_velocity_gradient,
                    stress_exponent=1.5,
                    deformation_exponent=3.5,
                    nucleation_efficiency=5,
                    gbm_mobility=0,
                    volume_fraction=1.0,
                )
                cosθ = np.cos(θ)
                cos2θ = np.cos(2 * θ)
                sinθ = np.sin(θ)
                target_orientations_diff = np.array(
                    [
                        [0, 0, 0],
                        [0, -sinθ * (1 + cos2θ), -cosθ * (1 + cos2θ)],
                        [0, cosθ * (1 + cos2θ), -sinθ * (1 + cos2θ)],
                    ]
                )
                np.testing.assert_allclose(
                    orientations_diff[0], target_orientations_diff
                )
                if outdir is not None:
                    initial_angles.append(np.rad2deg(θ))
                    rotation_rates.append(
                        np.sqrt(
                            orientations_diff[0][1, 1] ** 2
                            + orientations_diff[0][1, 2] ** 2
                        )
                    )
                    target_rotation_rates.append(1 + cos2θ)
                assert np.isclose(np.sum(fractions_diff), 0.0)

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 2 / 3))
            axs = fig.subplot_mosaic(  # Put the legend on a special axes.
                [["legend"], ["spin"]], gridspec_kw={"height_ratios": [0.1, 1]}
            )
            ax = axs["spin"]
            fig, ax, colors = _vis.spin(
                ax,
                initial_angles[::5],
                rotation_rates[::5],
                initial_angles,
                target_rotation_rates,
                shear_axis=90,
            )
            _utils.redraw_legend(
                ax,
                legendax=axs["legend"],
                loc="upper center",
                ncols=3,
                mode="expand",
            )
            fig.savefig(
                _io.resolve_path(f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}.pdf")
            )


class TestRecrystallisation2D:
    """Basic recrystallisation tests for 2D simple shear."""

    class_id = "recrystallisation_2D"

    def test_shear_dvdx_circle_inplane(self, outdir):
        r"""360000 grains of A-type olivine with uniform spread of a-axes on a circle.

        Grain growth rates are compared to analytical calculations.
        The a-axes are distributed in the YX plane (i.e.\ rotated around Z).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dvdx_circle_inplane"

        optional_logging = cl.nullcontext()
        # Initial uniform distribution of orientations on a 2D circle.
        initial_angles = np.mgrid[0 : 2 * np.pi : 360000j]
        cos2θ = np.cos(2 * initial_angles)
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")

        with optional_logging:
            initial_orientations = Rotation.from_rotvec(
                [[0, 0, θ] for θ in initial_angles]
            )
            orientations_diff, fractions_diff = _core.derivatives(
                phase=_core.MineralPhase.olivine,
                fabric=_core.MineralFabric.olivine_A,
                n_grains=360000,
                orientations=initial_orientations.as_matrix(),
                fractions=np.full(360000, 1 / 360000),
                strain_rate=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                velocity_gradient=np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]]),
                stress_exponent=1.5,
                deformation_exponent=3.5,
                nucleation_efficiency=5,
                gbm_mobility=125,
                volume_fraction=1.0,
            )
            ρ = np.abs(cos2θ) ** (1.5 / 3.5)
            # No need to sum over slip systems, only (010)[100] can activate in this
            # 2D simple shear deformation geometry (ρₛ=ρ).
            target_strain_energies = ρ * np.exp(-5 * ρ**2)
            target_fractions_diff = np.array(
                [  # df/dt* = - M* f (E - E_mean)
                    -125 * 1 / 360000 * (E - np.mean(target_strain_energies))
                    for E in target_strain_energies
                ]
            )

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 4 / 3))
            axs = fig.subplot_mosaic(  # Put the legend on a special axes.
                [["legend"], ["spin"], ["growth"]],
                gridspec_kw={"height_ratios": [0.2, 1, 1]},
                sharex=True,
            )
            ax = axs["spin"]
            xvals = np.rad2deg(initial_angles)
            fig, ax, colors = _vis.spin(
                ax,
                xvals[::2500],
                np.sqrt(
                    [
                        o[0, 0] ** 2 + o[0, 1] ** 2 + o[0, 2] ** 2
                        for o in orientations_diff
                    ][::2500]
                ),
                xvals,
                1 + cos2θ,
                shear_axis=90,
            )
            ax.get_legend().remove()
            ax.label_outer()
            ax2 = axs["growth"]
            fig, ax2, colors = _vis.growth(
                ax2,
                xvals[::2500],
                fractions_diff[::2500],
                xvals,
                target_fractions_diff,
                shear_axis=90,
            )
            _utils.redraw_legend(
                ax2,
                legendax=axs["legend"],
                loc="upper center",
                ncols=3,
                mode="expand",
            )
            _utils.add_subplot_labels(axs, labelmap={"spin": "a)", "growth": "b)"})
            fig.savefig(f"{out_basepath}.pdf")

        nt.assert_allclose(fractions_diff, target_fractions_diff, atol=1e-15, rtol=0)

    def test_shear_dvdx_circle_shearplane(self, outdir):
        r"""360000 grains of A-type olivine with uniform spread of a-axes on a circle.

        Unlike `test_shear_dvdx_circle_inplane`, two slip systems are active here,
        with cyclical variety in which one is dominant depending on grain orientation.
        The a-axes are distributed in the YZ plane
        (i.e.\ extrinsic rotation around Z by 90° and then around X).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        test_id = "dvdx_circle_shearplane"

        optional_logging = cl.nullcontext()
        # Initial uniform distribution of orientations on a 2D circle.
        initial_angles = np.mgrid[0 : 2 * np.pi : 360000j]
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_{test_id}"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")

        with optional_logging:
            initial_orientations = Rotation.from_euler(
                "zx", [[np.pi / 2, θ] for θ in initial_angles]
            )
            orientations_diff, fractions_diff = _core.derivatives(
                phase=_core.MineralPhase.olivine,
                fabric=_core.MineralFabric.olivine_A,
                n_grains=360000,
                orientations=initial_orientations.as_matrix(),
                fractions=np.full(360000, 1 / 360000),
                strain_rate=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                velocity_gradient=np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]]),
                stress_exponent=1.5,
                deformation_exponent=3.5,
                nucleation_efficiency=5,
                gbm_mobility=125,
                volume_fraction=1.0,
            )

        if outdir is not None:
            fig = _vis.figure(figscale=(1, 4 / 3))
            axs = fig.subplot_mosaic([["spin"], ["growth"]], sharex=True)
            ax = axs["spin"]
            xvals = np.rad2deg(initial_angles)
            fig, ax, colors = _vis.spin(
                ax,
                xvals[::2500],
                np.sqrt(
                    [
                        o[0, 0] ** 2 + o[0, 1] ** 2 + o[0, 2] ** 2
                        for o in orientations_diff
                    ][::2500]
                ),
            )
            ax.get_legend().remove()
            ax.label_outer()
            ax2 = axs["growth"]
            fig, ax2, colors = _vis.growth(ax2, xvals[::2500], fractions_diff[::2500])
            _utils.add_subplot_labels(axs, labelmap={"spin": "a)", "growth": "b)"})
            fig.savefig(f"{out_basepath}.pdf")

        # Check dominant slip system every 1°.
        for θ in initial_angles[::1000]:
            slip_invariants = _core._get_slip_invariants(
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                Rotation.from_euler("zx", [np.pi / 2, θ]).as_matrix(),
            )
            θ = np.rad2deg(θ)
            crss = _core.get_crss(
                _core.MineralPhase.olivine,
                _core.MineralFabric.olivine_A,
            )
            slip_indices = np.argsort(np.abs(slip_invariants / crss))
            slip_system = _minerals.OLIVINE_SLIP_SYSTEMS[slip_indices[-1]]

            if 0 <= θ < 64:
                assert slip_system == ([0, 1, 0], [1, 0, 0])
            elif 64 <= θ < 117:
                assert slip_system == ([0, 0, 1], [1, 0, 0])
            elif 117 <= θ < 244:
                assert slip_system == ([0, 1, 0], [1, 0, 0])
            elif 244 <= θ < 297:
                assert slip_system == ([0, 0, 1], [1, 0, 0])
            else:
                assert slip_system == ([0, 1, 0], [1, 0, 0])
