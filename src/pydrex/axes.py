"""PyDRex: Custom Matplotlib Axes subclasses."""
import numpy as np
import matplotlib.axes as mplax
import matplotlib as mpl
from matplotlib.projections import register_projection

from pydrex import stats as _stats
from pydrex import visualisation as _vis


class PoleFigureAxes(mplax.Axes):
    """Axes class designed for crystallographic pole figures.

    Thin matplotlib Axes wrapper for crystallographic pole figures.

    .. note:: Projections are not performed automatically using default methods like
        `scatter` or `plot`. To actually plot the pole figures, use `polefigure` method.

    """

    name = "pydrex.polefigure"

    def _prep_polefig_axis(self, ref_axes="xz"):
        """Set various options of a matplotlib `Axes` to prepare for a pole figure.

        Use a two-letter string for `ref_axes`.
        These letters will be used for the horizontal and vertical labels, respectively.

        """
        self.set_axis_off()
        self.set_xlim((-1.1, 1.1))
        self.set_ylim((-1.1, 1.1))
        self.set_aspect("equal")
        _circle_points = np.linspace(0, np.pi * 2, 100)
        self.plot(
            np.cos(_circle_points),
            np.sin(_circle_points),
            linewidth=1,
            color=mpl.rcParams["axes.edgecolor"],
        )
        self.axhline(0, color=mpl.rcParams["grid.color"], alpha=0.5)
        self.text(
            1.05, 0.5, ref_axes[0], verticalalignment="center", transform=self.transAxes
        )
        self.axvline(0, color=mpl.rcParams["grid.color"], alpha=0.5)

        self.text(
            0.5,
            1.05,
            ref_axes[1],
            horizontalalignment="center",
            transform=self.transAxes,
        )

    def polefigure(
        self,
        data,
        density=False,
        ref_axes="xz",
        hkl=[1, 0, 0],
        density_kwargs=None,
        **kwargs
    ):
        """Plot pole figure of crystallographic texture."""
        if density_kwargs is None:
            density_kwargs = {}

        self._prep_polefig_axis(ref_axes=ref_axes)

        if density:
            self.contourf(
                *_stats.point_density(
                    *_vis.poles(data, hkl=hkl, ref_axes=ref_axes), **density_kwargs
                ),
                **kwargs,
            )
        else:
            size = kwargs.pop("s", 0.3)
            alpha = kwargs.pop("alpha", 0.33)
            zorder = kwargs.pop("zorder", 11)
            self.scatter(
                *_vis.poles(data, hkl=hkl, ref_axes=ref_axes),
                s=size,
                alpha=alpha,
                zorder=zorder,
                **kwargs,
            )


register_projection(PoleFigureAxes)
