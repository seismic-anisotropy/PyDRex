"""> PyDRex: Mock objects for testing and reproducibility."""

from pydrex.core import MineralFabric, MineralPhase

PARAMS_FRATERS2021 = {
    "phase_assemblage": (MineralPhase.olivine, MineralPhase.enstatite),
    "phase_fractions": (0.7, 0.3),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 125,
    "gbs_threshold": 0.3,
    "nucleation_efficiency": 5,
    "number_of_grains": 5000,
}
"""Values used for tests 1, 2 and 4 in <https://doi.org/10.1029/2021gc009846>."""

PARAMS_KAMINSKI2001_FIG5_SOLID = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 0,
    "gbs_threshold": 0,
    "nucleation_efficiency": 5,
    "number_of_grains": 3375,  # 15^3 as in the Fortran code.
}
"""Values used for the M*=0 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>."""

PARAMS_KAMINSKI2001_FIG5_SHORTDASH = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 50,
    "gbs_threshold": 0,
    "nucleation_efficiency": 5,
    "number_of_grains": 3375,  # 15^3 as in the Fortran code.
}
"""Values used for the M*=50 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>."""

PARAMS_KAMINSKI2001_FIG5_LONGDASH = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 200,
    "gbs_threshold": 0,
    "nucleation_efficiency": 5,
    "number_of_grains": 3375,  # 15^3 as in the Fortran code.
}
"""Values used for the M*=200 test in <https://doi.org/10.1016/s0012-821x(01)00356-9>."""

PARAMS_KAMINSKI2004_FIG4_TRIANGLES = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 125,
    "gbs_threshold": 0.4,
    "nucleation_efficiency": 5,
    "number_of_grains": 4394,  # Section 4.1, first paragraph.
}
"""Values used for the χ=0.4 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

PARAMS_KAMINSKI2004_FIG4_SQUARES = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 125,
    "gbs_threshold": 0.2,
    "nucleation_efficiency": 5,
    "number_of_grains": 4394,  # Section 4.1, first paragraph.
}
"""Values used for the χ=0.2 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

PARAMS_KAMINSKI2004_FIG4_CIRCLES = {
    "phase_assemblage": (MineralPhase.olivine,),
    "phase_fractions": (1,),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 125,
    "gbs_threshold": 0,
    "nucleation_efficiency": 5,
    "number_of_grains": 4394,  # Section 4.1, first paragraph.
}
"""Values used for the χ=0 test in <https://doi.org/10.1111/j.1365-246x.2004.02308.x>."""

PARAMS_HEDJAZIAN2017 = {
    "phase_assemblage": (MineralPhase.olivine, MineralPhase.enstatite),
    "phase_fractions": (0.7, 0.3),
    "initial_olivine_fabric": MineralFabric.olivine_A,
    "stress_exponent": 1.5,
    "deformation_exponent": 3.5,
    "gbm_mobility": 10,
    "gbs_threshold": 0.2,
    "nucleation_efficiency": 5,
    "number_of_grains": 2197,  # 13^3 for both olivine and enstatite.
}
"""Values used for the MOR model in <https://doi.org/10.1016/j.epsl.2016.12.004>."""
