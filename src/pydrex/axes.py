"""> PyDRex: Custom Matplotlib Axes subclasses."""

import matplotlib as mpl
import matplotlib.axes as mplax
import numpy as np
from matplotlib.projections import register_projection

from pydrex import geometry as _geo
from pydrex import stats as _stats


class PoleFigureAxes(mplax.Axes):
    """Axes class designed for crystallographic pole figures.

    Thin matplotlib Axes wrapper for crystallographic pole figures.

    .. note::
        Projections are not performed automatically using default methods like
        `scatter` or `plot`. To actually plot the pole figures, use `polefigure`.

    """

    name = "pydrex.polefigure"

    def _prep_polefig_axis(self, ref_axes="xz"):
        """Set various options of a matplotlib `Axes` to prepare for a pole figure.

        Use a two-letter string for `ref_axes`.
        These letters will be used for the horizontal and vertical labels, respectively.

        """
        self.set_axis_off()
        self.set_aspect("equal")
        _circle_points = np.linspace(0, np.pi * 2, 100)
        self.plot(
            np.cos(_circle_points),
            np.sin(_circle_points),
            linewidth=0.25,
            color=mpl.rcParams["axes.edgecolor"],
        )
        self.axhline(0, color=mpl.rcParams["grid.color"], alpha=0.5)
        self.text(
            1.0,
            0.4,
            ref_axes[0],
            verticalalignment="center",
            transform=self.transAxes,
            fontsize="x-small",
        )
        self.axvline(0, color=mpl.rcParams["grid.color"], alpha=0.5)

        self.text(
            0.6,
            1.0,
            ref_axes[1],
            horizontalalignment="center",
            transform=self.transAxes,
            fontsize="x-small",
        )

    def polefigure(
        self,
        data,
        density=False,
        ref_axes="xz",
        hkl=[1, 0, 0],
        density_kwargs=None,
        **kwargs,
    ):
        """Plot pole figure of crystallographic texture.

        Args:
        - `data` (array) — Nx3x3 array of orientation matrices
        - `density` (bool, optional) — plot contoured pole figures, False by default
        - `ref_axes` (string, optional) — letters specifying the horizontal and vertical
          axes of the pole figure, and respective labels
        - `hkl` (array, optional) — crystallographic axis (one of the slip
          directions of olivine, i.e. [1, 0, 0], [0, 1, 0] or [0, 0, 1])
        - `density_kwargs` (dict, optional) — keyword arguments to pass to
          `pydrex.stats.point_density` if `density=True`

        Any additional keyword arguments are passed to either `tripcolor` if
        `density=True` or `scatter` if `density=False`

        """
        if density_kwargs is None:
            density_kwargs = {}

        self._prep_polefig_axis(ref_axes=ref_axes)

        if density:
            x, y, z = _stats.point_density(
                *_geo.poles(data, hkl=hkl, ref_axes=ref_axes), **density_kwargs
            )
            return self.tripcolor(
                x.ravel(),
                y.ravel(),
                z.ravel(),
                shading=kwargs.pop("shading", "gouraud"),
                **kwargs,
            )
        else:
            return self.scatter(
                *_geo.lambert_equal_area(
                    *_geo.poles(data, hkl=hkl, ref_axes=ref_axes),
                ),
                s=kwargs.pop("s", 1),
                c=kwargs.pop("c", mpl.rcParams["axes.edgecolor"]),
                marker=kwargs.pop("marker", "."),
                alpha=kwargs.pop("alpha", 0.33),
                zorder=kwargs.pop("zorder", 11),
                **kwargs,
            )


register_projection(PoleFigureAxes)
