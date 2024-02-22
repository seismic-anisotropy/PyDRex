"""> PyDRex: 2D simple shear tests."""

import contextlib as cl
import functools as ft
from multiprocessing import Pool
from time import process_time

import numpy as np
import pytest
from numpy import testing as nt
from scipy.interpolate import PchipInterpolator

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import utils as _utils
from pydrex import velocity as _velocity
from pydrex import visualisation as _vis

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_simple_shear"


class TestOlivineA:
    """Tests for stationary A-type olivine polycrystals in 2D simple shear."""

    class_id = "olivineA"

    @classmethod
    def get_position(cls, t):
        return np.zeros(3)  # These crystals are stationary.

    @classmethod
    def run(
        cls,
        params,
        timestamps,
        strain_rate,
        get_velocity_gradient,
        shear_direction,
        seed=None,
        return_fse=None,
    ):
        """Reusable logic for 2D olivine (A-type) simple shear tests.

        Returns a tuple with the mineral and the FSE angle (or `None` if `return_fse` is
        `None`).

        """
        mineral = _minerals.Mineral(
            phase=_core.MineralPhase.olivine,
            fabric=_core.MineralFabric.olivine_A,
            regime=_core.DeformationRegime.dislocation,
            n_grains=params["number_of_grains"],
            seed=seed,
        )
        deformation_gradient = np.eye(3)  # Undeformed initial state.
        θ_fse = np.empty_like(timestamps)
        θ_fse[0] = 45

        for t, time in enumerate(timestamps[:-1], start=1):
            # Set up logging message depending on dynamic parameter and seeds.
            msg_start = (
                f"N = {params['number_of_grains']}; "
                + f"λ∗ = {params['nucleation_efficiency']}; "
                + f"X = {params['gbs_threshold']}; "
                + f"M∗ = {params['gbm_mobility']}; "
            )
            if seed is not None:
                msg_start += f"# {seed}; "

            _log.info(msg_start + "step %s/%s (t = %s)", t, len(timestamps) - 1, time)

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], cls.get_position),
            )
            _log.debug(
                "› velocity gradient = %s",
                get_velocity_gradient(None).flatten(),
            )
            _log.debug("› strain D₀t = %.2f", strain_rate * timestamps[t])
            _log.debug(
                "› grain fractions: median = %s, max = %s, min = %s",
                np.median(mineral.fractions[-1]),
                np.max(mineral.fractions[-1]),
                np.min(mineral.fractions[-1]),
            )
            if return_fse:
                _, fse_v = _diagnostics.finite_strain(deformation_gradient)
                θ_fse[t] = _diagnostics.smallest_angle(fse_v, shear_direction)
            else:
                θ_fse = None

        return mineral, θ_fse

    @classmethod
    def interp_GBM_Kaminski2001(cls, strains):
        """Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`."""
        _log.info("interpolating target CPO angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2001_GBMshear.scsv")
        cs_M0 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M0) / 200,
            _utils.remove_nans(data.angle_M0),
        )
        cs_M50 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M50) / 200,
            _utils.remove_nans(data.angle_M50),
        )
        cs_M200 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M200) / 200,
            _utils.remove_nans(data.angle_M200),
        )
        return [cs_M0(strains), cs_M50(strains), cs_M200(strains)]

    @classmethod
    def interp_GBM_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("drexF90") / "olA_D1E4_dt50_X0_L5.scsv")
        data_strains = np.linspace(0, 1, 200)
        cs_M0 = PchipInterpolator(data_strains, _utils.remove_nans(data.M0_angle))
        cs_M50 = PchipInterpolator(data_strains, _utils.remove_nans(data.M50_angle))
        cs_M200 = PchipInterpolator(data_strains, _utils.remove_nans(data.M200_angle))
        return [cs_M0(strains), cs_M50(strains), cs_M200(strains)]

    @classmethod
    def interp_GBS_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "a_axis_GBS_fortran.scsv")
        data_strains = np.linspace(0, 1, 200)
        cs_X0 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X0))
        cs_X20 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X20))
        cs_X40 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X40))
        return [cs_X0(strains), cs_X20(strains), cs_X40(strains)]

    @classmethod
    def interp_GBS_long_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "a_axis_GBS_long_fortran.scsv")
        data_strains = np.linspace(0, 2.5, 500)
        cs_X0 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X0))
        cs_X20 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X20))
        cs_X40 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X40))
        return [cs_X0(strains), cs_X20(strains), cs_X40(strains)]

    @classmethod
    def interp_GBS_Kaminski2004(cls, strains):
        """Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`."""
        _log.info("interpolating target CPO angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2004_GBSshear.scsv")
        cs_X0 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0),
            45 + _utils.remove_nans(data.angle_X0),
        )
        cs_X0d2 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0d2),
            45 + _utils.remove_nans(data.angle_X0d2),
        )
        cs_X0d4 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0d4),
            45 + _utils.remove_nans(data.angle_X0d4),
        )
        return [cs_X0(strains), cs_X0d2(strains), cs_X0d4(strains)]

    def test_zero_recrystallisation(self, seed):
        """Check that M*=0 is a reliable switch to turn off recrystallisation."""
        params = _io.DEFAULT_PARAMS
        params["gbm_mobility"] = 0
        strain_rate = 1
        timestamps = np.linspace(0, 1, 25)  # Solve until D₀t=1 (tensorial strain).
        shear_direction = [0, 1, 0]
        _, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
        mineral, _ = self.run(
            params,
            timestamps,
            strain_rate,
            get_velocity_gradient,
            shear_direction,
            seed=seed,
        )
        for fractions in mineral.fractions[1:]:
            nt.assert_allclose(fractions, mineral.fractions[0], atol=1e-15, rtol=0)

    @pytest.mark.parametrize("gbm_mobility", [50, 100, 150])
    def test_grainsize_median(self, seed, gbm_mobility):
        """Check that M*={50,100,150}, λ*=5 causes decreasing grain size median."""
        params = _io.DEFAULT_PARAMS
        params["gbm_mobility"] = gbm_mobility
        params["nucleation_efficiency"] = 5
        strain_rate = 1
        timestamps = np.linspace(0, 1, 25)  # Solve until D₀t=1 (tensorial strain).
        n_timestamps = len(timestamps)
        shear_direction = [0, 1, 0]
        _, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
        mineral, _ = self.run(
            params,
            timestamps,
            strain_rate,
            get_velocity_gradient,
            shear_direction,
            seed=seed,
        )
        medians = np.empty(n_timestamps)
        for i, fractions in enumerate(mineral.fractions):
            medians[i] = np.median(fractions)

        # The first diff is positive (~1e-6) before nucleation sets in.
        nt.assert_array_less(np.diff(medians)[1:], np.full(n_timestamps - 2, 0))

    @pytest.mark.slow
    @pytest.mark.parametrize("gbs_threshold", [0, 0.2, 0.4])
    @pytest.mark.parametrize("nucleation_efficiency", [3, 5, 10])
    def test_dvdx_ensemble(
        self, outdir, seeds_nearX45, ncpus, gbs_threshold, nucleation_efficiency
    ):
        r"""Test a-axis alignment to shear in Y direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        strain_rate = 1
        timestamps = np.linspace(0, 1, 201)  # Solve until D₀t=1 ('shear' γ=2).
        n_timestamps = len(timestamps)
        # Use `seeds` instead of `seeds_nearX45` if you have even more RAM and CPU time.
        _seeds = seeds_nearX45
        n_seeds = len(_seeds)

        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        _, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)

        gbm_mobilities = [0, 50, 125, 200]
        markers = ("x", "*", "d", "s")

        _id = f"X{_io.stringify(gbs_threshold)}_L{_io.stringify(nucleation_efficiency)}"
        # Output setup with optional logging and data series labels.
        θ_fse = np.empty_like(timestamps)
        angles = np.empty((len(gbm_mobilities), n_seeds, n_timestamps))
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dvdx_ensemble_{_id}"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            clock_start = process_time()
            for m, gbm_mobility in enumerate(gbm_mobilities):
                if m == 0:
                    return_fse = True
                else:
                    return_fse = False

                params = {
                    "olivine_fraction": 1.0,
                    "enstatite_fraction": 0.0,
                    "stress_exponent": 1.5,
                    "deformation_exponent": 3.5,
                    "gbm_mobility": gbm_mobility,
                    "gbs_threshold": gbs_threshold,
                    "nucleation_efficiency": nucleation_efficiency,
                    "number_of_grains": 5000,
                    "initial_olivine_fabric": "A",
                }

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    return_fse=return_fse,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                        mineral, fse_angles = out
                        angles[m, s, :] = [
                            _diagnostics.smallest_angle(v, shear_direction)
                            for v in _diagnostics.elasticity_components(
                                _minerals.voigt_averages([mineral], params)
                            )["hexagonal_axis"]
                        ]
                        # Save the whole mineral for the first seed only.
                        if outdir is not None and s == 0:
                            postfix = (
                                f"M{_io.stringify(gbm_mobility)}"
                                + f"_X{_io.stringify(gbs_threshold)}"
                                + f"_L{_io.stringify(nucleation_efficiency)}"
                            )
                            mineral.save(f"{out_basepath}.npz", postfix=postfix)
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= n_seeds

                if outdir is not None:
                    labels.append(f"$M^∗$ = {gbm_mobility}")

            _log.info("elapsed CPU time: %s", np.abs(process_time() - clock_start))

        # Take ensemble means and optionally plot figure.
        strains = timestamps * strain_rate
        _log.info("postprocessing results for %s", _id)
        result_angles = angles.mean(axis=1)
        result_angles_err = angles.std(axis=1)

        if outdir is not None:
            schema = {
                "delimiter": ",",
                "missing": "-",
                "fields": [
                    {
                        "name": "strain",
                        "type": "integer",
                        "unit": "percent",
                        "fill": 999999,
                    }
                ],
            }
            np.savez(
                f"{out_basepath}.npz",
                angles=result_angles,
                angles_err=result_angles_err,
            )
            _io.save_scsv(
                f"{out_basepath}_strains.scsv",
                schema,
                [[int(D * 200) for D in strains]],  # Shear strain % is 200 * D₀.
            )
            fig, ax, colors = _vis.alignment(
                None,
                strains,
                result_angles,
                markers,
                labels,
                err=result_angles_err,
                θ_max=60,
                θ_fse=θ_fse,
            )
            fig.savefig(_io.resolve_path(f"{out_basepath}.pdf"))

    @pytest.mark.slow
    def test_dvdx_GBM(self, outdir, seeds_nearX45, ncpus):
        r"""Test a-axis alignment to shear in Y direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$
        \bm{L} = 10^{-4} ×
            \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Results are compared to the Fortran 90 output.

        """
        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        strain_rate = 1e-4
        _, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
        timestamps = np.linspace(0, 1e4, 51)  # Solve until D₀t=1 ('shear' γ=2).
        i_strain_40p = 10  # Index of 40% strain, lower strains are not relevant here.
        i_strain_100p = 25  # Index of 100% strain, when M*=0 matches FSE.
        params = _io.DEFAULT_PARAMS
        params["gbs_threshold"] = 0  # No GBS, to match the Fortran parameters.
        gbm_mobilities = (0, 10, 50, 125, 200)  # Must be in ascending order.
        markers = ("x", ".", "*", "d", "s")
        # Use `seeds` instead of `seeds_nearX45` if you have even more RAM and CPU time.
        _seeds = seeds_nearX45
        n_seeds = len(_seeds)
        angles = np.empty((len(gbm_mobilities), n_seeds, len(timestamps)))
        θ_fse = np.zeros_like(timestamps)
        strains = timestamps * strain_rate
        M0_drexF90, M50_drexF90, M200_drexF90 = self.interp_GBM_FortranDRex(strains)

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_mobility"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            clock_start = process_time()
            for m, gbm_mobility in enumerate(gbm_mobilities):
                if m == 0:
                    return_fse = True
                else:
                    return_fse = False
                params["gbm_mobility"] = gbm_mobility

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    return_fse=True,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                        mineral, fse_angles = out
                        angles[m, s, :] = [
                            _diagnostics.smallest_angle(v, shear_direction)
                            for v in _diagnostics.elasticity_components(
                                _minerals.voigt_averages([mineral], params)
                            )["hexagonal_axis"]
                        ]
                        # Save the whole mineral for the first seed only.
                        if outdir is not None and s == 0:
                            mineral.save(
                                f"{out_basepath}.npz",
                                postfix=f"M{_io.stringify(gbm_mobility)}",
                            )
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= n_seeds

                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")

            _log.info("elapsed CPU time: %s", np.abs(process_time() - clock_start))

            # Take ensemble means and optionally plot figure.
            _log.info("postprocessing results...")
            result_angles = angles.mean(axis=1)
            result_angles_err = angles.std(axis=1)

            if outdir is not None:
                schema = {
                    "delimiter": ",",
                    "missing": "-",
                    "fields": [
                        {
                            "name": "strain",
                            "type": "integer",
                            "unit": "percent",
                            "fill": 999999,
                        }
                    ],
                }
                _io.save_scsv(
                    f"{out_basepath}_strains.scsv",
                    schema,
                    [[int(D * 200) for D in strains]],  # Shear strain % is 200 * D₀.
                )
                np.savez(
                    f"{out_basepath}_angles.npz",
                    angles=result_angles,
                    err=result_angles_err,
                )
                fig, ax, colors = _vis.alignment(
                    None,
                    strains,
                    result_angles,
                    markers,
                    labels,
                    err=result_angles_err,
                    θ_max=60,
                    θ_fse=θ_fse,
                )
                ax.plot(strains, M0_drexF90, c=colors[0])
                ax.plot(strains, M50_drexF90, c=colors[2])
                ax.plot(strains, M200_drexF90, c=colors[4])
                _vis.show_Skemer2016_ShearStrainAngles(
                    ax,
                    ["Z&K 1200 C", "Z&K 1300 C"],
                    ["v", "^"],
                    ["k", "k"],
                    ["none", None],
                    [
                        "Zhang & Karato, 1995\n(1473 K)",
                        "Zhang & Karato, 1995\n(1573 K)",
                    ],
                    _core.MineralFabric.olivine_A,
                )
                # There is a lot of stuff on this legend, so put it outside the axes.
                # These values might need to be tweaked depending on the font size, etc.
                _legend = _utils.redraw_legend(ax, fig=fig, bbox_to_anchor=(1.66, 0.99))
                fig.savefig(
                    _io.resolve_path(f"{out_basepath}.pdf"),
                    bbox_extra_artists=(_legend,),
                    bbox_inches="tight",
                )

            # Check that GBM speeds up the alignment between 40% and 100% strain.
            _log.info("checking grain orientations...")
            for i, θ in enumerate(result_angles[:-1], start=1):
                nt.assert_array_less(
                    result_angles[i][i_strain_40p:i_strain_100p],
                    θ[i_strain_40p:i_strain_100p],
                )

            # Check that M*=0 matches FSE (±1°) past 100% strain.
            nt.assert_allclose(
                result_angles[0][i_strain_100p:],
                θ_fse[i_strain_100p:],
                atol=1,
                rtol=0,
            )

            # Check that results match Fortran output past 40% strain.
            nt.assert_allclose(
                result_angles[0][i_strain_40p:],
                M0_drexF90[i_strain_40p:],
                atol=0,
                rtol=0.1,  # At 40% strain the match is worse than at higher strain.
            )
            nt.assert_allclose(
                result_angles[2][i_strain_40p:],
                M50_drexF90[i_strain_40p:],
                atol=1,
                rtol=0,
            )
            nt.assert_allclose(
                result_angles[4][i_strain_40p:],
                M200_drexF90[i_strain_40p:],
                atol=1.5,
                rtol=0,
            )

    @pytest.mark.slow
    def test_GBM_calibration(self, outdir, seeds, ncpus):
        r"""Compare results for various values of $$M^∗$$ to A-type olivine data.

        Velocity gradient:
        $$
        \bm{L} = 10^{-4} ×
            \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Unlike `test_dvdx_GBM`,
        grain boudary sliding is enabled here (see `_io.DEFAULT_PARAMS`).
        Data are provided by [Skemer & Hansen, 2016](http://dx.doi.org/10.1016/j.tecto.2015.12.003).

        """
        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        strain_rate = 1
        _, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
        timestamps = np.linspace(0, 3.2, 65)  # Solve until D₀t=3.2 ('shear' γ=6.4).
        params = _io.DEFAULT_PARAMS
        params["number_of_grains"] = 5000
        gbm_mobilities = (0, 10, 50, 125)  # Must be in ascending order.
        markers = ("x", "*", "1", ".")
        # Uses 100 seeds by default; use all 1000 if you have more RAM and CPU time.
        _seeds = seeds[:100]
        n_seeds = len(_seeds)
        angles = np.empty((len(gbm_mobilities), n_seeds, len(timestamps)))
        θ_fse = np.zeros_like(timestamps)
        strains = timestamps * strain_rate

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_calibration"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            clock_start = process_time()
            for m, gbm_mobility in enumerate(gbm_mobilities):
                return_fse = True if m == 0 else False
                params["gbm_mobility"] = gbm_mobility
                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    return_fse=return_fse,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                        mineral, fse_angles = out
                        angles[m, s, :] = [
                            _diagnostics.smallest_angle(v, shear_direction)
                            for v in _diagnostics.elasticity_components(
                                _minerals.voigt_averages([mineral], params)
                            )["hexagonal_axis"]
                        ]
                        # Save the whole mineral for the first seed only.
                        if outdir is not None and s == 0:
                            mineral.save(
                                f"{out_basepath}.npz",
                                postfix=f"M{_io.stringify(gbm_mobility)}",
                            )
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= n_seeds

                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")

            _log.info("elapsed CPU time: %s", np.abs(process_time() - clock_start))

            # Take ensemble means and optionally plot figure.
            _log.info("postprocessing results...")
            result_angles = angles.mean(axis=1)
            result_angles_err = angles.std(axis=1)

            if outdir is not None:
                schema = {
                    "delimiter": ",",
                    "missing": "-",
                    "fields": [
                        {
                            "name": "strain",
                            "type": "integer",
                            "unit": "percent",
                            "fill": 999999,
                        }
                    ],
                }
                _io.save_scsv(
                    f"{out_basepath}_strains.scsv",
                    schema,
                    [[int(D * 200) for D in strains]],  # Shear strain % is 200 * D₀.
                )
                np.savez(
                    _io.resolve_path(f"{out_basepath}_ensemble_means.npz"),
                    angles=result_angles,
                    err=result_angles_err,
                )
                fig = _vis.figure(
                    figsize=(_vis.DEFAULT_FIG_WIDTH * 3, _vis.DEFAULT_FIG_HEIGHT)
                )
                fig, ax, colors = _vis.alignment(
                    fig.add_subplot(),
                    strains,
                    result_angles,
                    markers,
                    labels,
                    err=result_angles_err,
                    θ_max=80,
                    θ_fse=θ_fse,
                )
                (
                    _,
                    _,
                    _,
                    data_Skemer2016,
                    indices,
                ) = _vis.show_Skemer2016_ShearStrainAngles(
                    ax,
                    [
                        "Z&K 1200 C",
                        "Z&K 1300 C",
                        "Skemer 2011",
                        "Hansen 2014",
                        "Warren 2008",
                        "Webber 2010",
                        "H&W 2015",
                    ],
                    ["v", "^", "o", "s", "v", "o", "s"],
                    ["k", "k", "k", "k", "k", "k", "k"],
                    ["none", "none", "none", "none", None, None, None],
                    [
                        "Zhang & Karato, 1995 (1473 K)",
                        "Zhang & Karato, 1995 (1573 K)",
                        "Skemer et al., 2011 (1500 K)",
                        "Hansen et al., 2014 (1473 K)",
                        "Warren et al., 2008",
                        "Webber et al., 2010",
                        "Hansen & Warren, 2015",
                    ],
                    fabric=_core.MineralFabric.olivine_A,
                )
                _legend = _utils.redraw_legend(ax, loc="upper right", ncols=3)
                fig.savefig(
                    _io.resolve_path(f"{out_basepath}.pdf"),
                    bbox_extra_artists=(_legend,),
                    bbox_inches="tight",
                )
                r2vals = []
                for angles in result_angles:
                    _angles = PchipInterpolator(strains, angles)
                    r2 = np.sum(
                        [
                            (a - b) ** 2
                            for a, b in zip(
                                _angles(
                                    np.take(data_Skemer2016.shear_strain, indices) / 200
                                ),
                                np.take(data_Skemer2016.angle, indices),
                            )
                        ]
                    )
                    r2vals.append(r2)
                _log.info(
                    "Sums of squared residuals (r-values) for each M∗: %s", r2vals
                )
