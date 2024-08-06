! Fortran 2003 implementations of selected core PyDRex routines.
! These are intended for performance comparisons.
module core
    use iso_fortran_env, only: real64
    use ieee_arithmetic, only: inf
    use misc
    implicit none
    contains

pure function crss(phase, fabric)
    ! Get Critical Resolved Shear Stress for the mineral `phase` and `fabric`.
    integer, intent(in) :: phase, fabric
    real(kind=real64), dimension(4) :: crss

    if phase == 0 then  ! olivine
        if fabric == 0 then  ! A-type
            crss = [1.0, 2.0, 3.0, inf]
        else if fabric == 1 then  ! B-type
            crss = [3.0, 2.0, 1.0, inf]
        else if fabric == 2 then  ! C-type
            crss = [3.0, 2.0, inf, 1.0]
        else if fabric == 3 then  ! D-type
            crss = [1.0, 1.0, 3.0, inf]
        else if fabric == 4 then  ! E-type
            crss = [3.0, 1.0, 2.0, inf]
        end if
    else if phase == 1 then  ! enstatite
        crss = [inf, inf, inf, 1]
        ! TODO: Throw error if fabric not equal to 5.
    end if
    ! TODO: Else throw error.
end function crss

pure function slip_invariants(strain_rate, orientation)
    ! Calculate strain rate invariants for minerals with four slip systems.
    real(kind=real64), intent(in), dimension(3,3) :: strain_rate, orientation
    real(kind=real64), dimension(4) :: slip_invariants

    integer :: i, j
    do i = 1, 3
        do j = 1, 3
            slip_invariants(1) = slip_invariants(1) &
                + strain_rate(i, j) * orientation(1, i) + orientation(2, j)
            slip_invariants(2) = slip_invariants(2) &
                + strain_rate(i, j) * orientation(1, i) + orientation(3, j)
            slip_invariants(3) = slip_invariants(3) &
                + strain_rate(i, j) * orientation(3, i) + orientation(2, j)
            slip_invariants(4) = slip_invariants(4) &
                + strain_rate(i, j) * orientation(3, i) + orientation(1, j)
        end do
    end do
end function slip_invariants

pure function slip_rates_olivine(invariants, slip_indices, ol_crss, deformation_exponent)
    ! Calculate relative slip rates of the active slip systems for olivine.
    real(kind=real64), intent(in), dimension(4) :: invariants, ol_crss
    integer, intent(in), dimension(4) :: slip_indices
    real(kind=real64), intent(in) :: deformation_exponent

    real(kind=real64), dimension(4) :: slip_rates_olivine

    real(kind=real64) :: i_inac, i_min, i_int, i_max
    i_inac = slip_indices(1)
    i_min = slip_indices(2)
    i_int = slip_indices(3)
    i_max = slip_indices(4)

    prefactor = ol_crss(i_max) / invariants(i_max)
    ratio_min = prefactor * invariants(i_min) / ol_crss(i_min)
    ratio_int = prefactor * invariants(i_int) / ol_crss(i_int)
    slip_rates_olivine(i_inac) = 0  ! Hardest system is completely inactive in olivine.
    slip_rates_olivine(i_min) = ratio_min * np.abs(ratio_min) ** (deformation_exponent - 1)
    slip_rates_olivine(i_int) = ratio_int * np.abs(ratio_int) ** (deformation_exponent - 1)
    slip_rates_olivine(i_max) = 1
end function slip_rates_olivine

pure function rate_of_deformation(phase, orientation, slip_rates)
    ! Calculate deformation rate tensor for olivine or enstatite.
    integer, intent(in) :: phase
    real(kind=real64), intent(in), dimension(3,3) :: orientation
    real(kind=real64), intent(in), dimension(4) :: slip_rates

    real(kind=real64), dimension(3,3) :: rate_of_deformation

    integer :: i, j
    do i = 1, 3
        do j = 1, 3
            rate_of_deformation(i, j) = 2 * (
                slip_rates(1) * orientation(1, i) * orientation(2, j) &
                + slip_rates(2) * orientation(1, i) * orientation(3, j) &
                + slip_rates(3) * orientation(3, i) * orientation(2, j) &
                + slip_rates(4) * orientation(3, i) * orientation(1, j) &
            )
        end do
    end do
end function rate_of_deformation

pure function slip_rate_softest(deformation_rate, velocity_gradient)
    ! Calculate dimensionless strain rate on the softest slip system.
    real(kind=real64), intent(in), dimension(3,3) :: deformation_rate, velocity_gradient
    real(kind=real64) :: slip_rate_softest

    real(kind=real64) :: enumerator, denominator

    integer :: j, L
    do j = 1, 3
        k = (j + 1) % 3  ! Using correction from Fraters & Billen 2021 S1.
        enumerator = enumerator - (velocity_gradient(j, k) - velocity_gradient(k, j)) * &
            (deformation_rate(j, k) - deformation_rate(k, j))
        ! Using correction from Fraters & Billen 2021 S1.
        denominator = denominator - (deformation_rate(j, k) - deformation_rate(k, j)) ** 2

        do L = 1, 3
            enumerator = enumerator + 2 * deformation_rate(j, L) * velocity_gradient(j, L)
            denominator = denominator + 2 * deformation_rate(j, L) ** 2
        end do
    end do

    ! Avoid zero division.
    if -1e-15 < denominator .and. denominator < 1e-15 then
        slip_rate_softest = 0e0
    end if
    slip_rate_softest = enumerator / denominator
end function get_slip_rate_softest

pure function orientation_change( &
    orientation, &
    velocity_gradient, &
    deformation_rate, &
    softest_rate, &
    permutation_symbol, &
)
    ! Calculate the rotation for a grain undergoing dislocation creep.
    real(kind=real64), intent(in), dimension(3,3) :: orientation
    real(kind=real64), intent(in), dimension(3,3) :: deformation_rate, velocity_gradient
    real(kind=real64), intent(in), dimension(3,3,3) :: permutation_symbol
    real(kind=real64), intent(in) :: softest_rate

    real(kind=real64), dimension(3, 3) :: orientation_change

    real(kind=real64), dimension(3) :: spin_vector

    integer :: j, p, q, r, s
    do j = 1, 3
        r = (j + 1) % 3
        s = (j + 2) % 3
        spin_vector(j) = ((velocity_gradient(s, r) - velocity_gradient(r, s)) &
            - (deformation_rate(s, r) - deformation_rate(r, s)) * softest_rate) / 2
    end do

    do p = 1, 3
        do q = 1, 3
            do r = 1, 3
                do s = 1, 3
                    orientation_change(p, q) = orientation_change(p, q) &
                        + permutation_symbol(q, r, s) * orientation(p, s) * spin_vector(r)
                end do
            end do
        end do
    end do
end function orientation_change

pure function strain_energy( &
    ol_crss, &
    slip_rates, &
    slip_indices, &
    softest_rate, &
    stress_exponent, &
    deformation_exponent, &
    nucleation_efficiency, &
)
    ! Calculate strain energy due to dislocations for an olivine grain.
    real(kind=real64), intent(in), dimension(4) :: ol_crss, slip_rates
    integer, intent(in), dimension(4) :: slip_indices
    real(kind=real64), intent(in) :: softest_rate, nucleation_efficiency
    real(kind=real64), intent(in) :: stress_exponent, deformation_exponent

    real(kind=real64) :: strain_energy

    ! Corrected for spurrious division by strain rate scale in Eq. 11, Kaminski 2004.
    integer :: i
    real(kind=real64) :: dislocation_density
    strain_energy = 0.0
    do i = 1, 3
        dislocation_density = (1 / ol_crss(i)) ** (deformation_exponent - stress_exponent) &
            * np.abs(slip_rates(i) * softest_rate) &
            ** (stress_exponent / deformation_exponent)
        ! Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
        strain_energy = strain_energy + dislocation_density &
            * np.exp(-nucleation_efficiency * dislocation_density**2)
    end do
end function get_strain_energy

subroutine get_rotation_and_strain( &
    phase, &
    fabric, &
    orientation, &
    strain_rate, &
    velocity_gradient, &
    stress_exponent, &
    deformation_exponent, &
    nucleation_efficiency, &
    permutation_symbol, &
    out_spin, &
    out_energy, &
)
    ! Get the crystal axes rotation rate and strain energy of an individual grain.
    ! Arguments and returned values as per pydrex.core._get_rotation_and_strain().
    ! Only implemented for the olivine A-type mineral phase.
    integer, intent(in) :: phase, fabric
    real(kind=real64), intent(in) :: nucleation_efficiency
    real(kind=real64), intent(in) :: stress_exponent, deformation_exponent
    real(kind=real64), intent(in), dimension(3,3) :: strain_rate, velocity_gradient
    real(kind=real64), intent(in), dimension(3,3) :: orientation
    real(kind=real64), intent(in), dimension(3,3,3) :: permutation_symbol

    real(kind=real64), intent(out), dimension(3,3) :: out_spin
    real(kind=real64), intent(out) :: out_energy

    real(kind=real64), dimension(4) :: _crss, _slip_invariants, _abs_array
    real(kind=real64) :: _deformation_rate, _slip_rate_softest

    integer :: i
    _crss = crss(phase, fabric)
    _slip_invariants = slip_invariants(strain_rate, orientation)
    do i = 1, 4
        _abs_array(i) = abs(_slip_invariants(i) / _crss(i))
    end do
    if phase == 0 then  ! olivine
        slip_indices = argsort(_abs_array)
        slip_rates = slip_rates_olivine(
            _slip_invariants, slip_indices, _crss, deformation_exponent
        )
    else if phase == 1 then  ! enstatite
        do i = 1, 4 then
            _abs_array(i) = 1 / _crss(i)
        end do
        slip_indices = argsort(_abs_array)
        slip_rates = [0.0, 0.0, 0.0, 0.0]
        if abs(_slip_invariants(size(_slip_invariants))) > 1e-15 then
            slip_rates(size(slip_rates)) = 1
        end if
    end if

    _deformation_rate = rate_of_deformation(phase, orientation, slip_rates)
    _slip_rate_softest = slip_rate_softest(_deformation_rate, velocity_gradient)
    out_orientation_change = orientation_change( &
        orientation, &
        velocity_gradient, &
        _deformation_rate, &
        _slip_rate_softest, &
        permutation_symbol, &
    )
    out_energy = strain_energy( &
        _crss, &
        slip_rates, &
        slip_indices, &
        _slip_rate_softest, &
        stress_exponent, &
        deformation_exponent, &
        nucleation_efficiency, &
    )
end subroutine get_rotation_and_strain

subroutine derivatives( &
    regime, &
    phase, &
    fabric, &
    n_grains, &
    orientations, &
    fractions, &
    strain_rate, &
    velocity_gradient, &
    deformation_gradient_spin, &
    stress_exponent, &
    deformation_exponent, &
    nucleation_efficiency, &
    gbm_mobility, &
    volume_fraction, &
    permutation_symbol, &
    out_spin, &
    out_growth, &
)
    ! Get derivatives of orientation and volume distribution.
    ! Arguments and returned values as per pydrex.core.derivatives().
    ! Only implemented for the matrix dislocation creep regime.
    integer, intent(in) :: regime, phase, fabric, n_grains
    real(kind=real64), intent(in) :: stress_exponent, deformation_exponent
    real(kind=real64), intent(in) :: nucleation_efficiency, gbm_mobility, volume_fraction
    real(kind=real64), intent(in), dimension(3,3) :: strain_rate, velocity_gradient
    real(kind=real64), intent(in), dimension(3,3) :: deformation_gradient_spin
    real(kind=real64), intent(in), dimension(n_grains,3,3) :: orientations
    real(kind=real64), intent(in), dimension(n_grains) :: fractions
    real(kind=real64), intent(in), dimension(3,3,3) :: permutation_symbol

    real(kind=real64), intent(out), dimension(n_grains,3,3) :: out_spin
    real(kind=real64), intent(out), dimension(n_grains) :: out_growth

    real(kind=real64), dimension(n_grains) :: _energies
    real(kind=real64), dimension(3,3) :: _orientation_change
    real(kind=real64) :: _energy, _mean_energy

    integer :: grain_index
    if regime == 4 then  ! matrix dislocation creep
        do grain_index = 1, n_grains
            call get_rotation_and_strain( &
                phase, &
                fabric, &
                orientations(grain_index), &
                strain_rate, &
                velocity_gradient, &
                stress_exponent, &
                deformation_exponent, &
                nucleation_efficiency, &
                _orientation_change, &
                _energy, &
            )
            out_spin(grain_index) = _orientation_change
            _energies(grain_index) = fractions(grain_index) * _energy
        end do

        _mean_energy = sum(_energies)
        do grain_index = 1, n_grains
            out_growth(grain_index) = volume_fraction * gbm_mobility &
                * fractions(grain_index) * _mean_energy - _energies(grain_index)
        end do
    end if
    ! TODO: Else throw error?
end subroutine derivatives

end module core
