"""> PyDRex: Mock objects for testing and reproducibility."""

from pydrex.core import DefaultParams, MineralFabric, MineralPhase


class ParamsFraters2021(DefaultParams):
    """Values used for tests 1, 2 and 4 in <https://doi.org/10.1029/2021gc009846>."""

    phase_assemblage = (MineralPhase.olivine, MineralPhase.enstatite)
    phase_fractions = (0.7, 0.3)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 125
    gbs_threshold = 0.3
    nucleation_efficiency = 5.0
    number_of_grains = 5000


class ParamsKaminski2001_Fig5Solid(DefaultParams):
    """Values used for the M*=0 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>."""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 0
    gbs_threshold = 0
    nucleation_efficiency = 5
    number_of_grains = 3375  # 15^3 as in the Fortran code.


class ParamsKaminski2001_Fig5ShortDash(DefaultParams):
    """Values used for the M*=50 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>."""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 50
    gbs_threshold = 0
    nucleation_efficiency = 5
    number_of_grains = 3375  # 15^3 as in the Fortran code.


class ParamsKaminski2001_Fig5LongDash(DefaultParams):
    """Values used for the M*=200 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>"""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 200
    gbs_threshold = 0
    nucleation_efficiency = 5
    number_of_grains = 3375  # 15^3 as in the Fortran code.


class ParamsKaminski2004_Fig4Triangles(DefaultParams):
    """Values used for the χ=0.4 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 125
    gbs_threshold = 0.4
    nucleation_efficiency = 5
    number_of_grains = 4394  # Section 4.1, first paragraph.


class ParamsKaminski2004_Fig4Squares(DefaultParams):
    """Values used for the χ=0.2 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 125
    gbs_threshold = 0.2
    nucleation_efficiency = 5
    number_of_grains = 4394  # Section 4.1, first paragraph.


class ParamsKaminski2004_Fig4Circles(DefaultParams):
    """Values used for the χ=0 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

    phase_assemblage = (MineralPhase.olivine,)
    phase_fractions = (1,)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 125
    gbs_threshold = 0
    nucleation_efficiency = 5
    number_of_grains = 4394  # Section 4.1, first paragraph.


class ParamsHedjazian2017(DefaultParams):
    """Values used for the MOR model in <https://doi.org/10.1016/j.epsl.2016.12.004>."""

    phase_assemblage = (MineralPhase.olivine, MineralPhase.enstatite)
    phase_fractions = (0.7, 0.3)
    initial_olivine_fabric = MineralFabric.olivine_A
    stress_exponent = 1.5
    deformation_exponent = 3.5
    gbm_mobility = 10
    gbs_threshold = 0.2
    nucleation_efficiency = 5
    number_of_grains = 2197  # 13^3 for both olivine and enstatite.
