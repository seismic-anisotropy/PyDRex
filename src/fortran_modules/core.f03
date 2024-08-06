! Fortran 2003 implementations of selected core PyDRex routines.
! These are intended for performance comparisons.
module core
    use iso_fortran_env, only: real64
    use ieee_arithmetic, only: ieee_value, ieee_positive_inf
    use misc
    implicit none
    contains

pure function get_crss(phase, fabric)
    ! Get Critical Resolved Shear Stress for the mineral `phase` and `fabric`.
    integer, intent(in) :: phase, fabric
    real(kind=real64), dimension(4) :: get_crss
    real(kind=real64) :: inf
    inf = ieee_value(inf, ieee_positive_inf)

    if (phase == 0) then  ! olivine
        if (fabric == 0) then  ! A-type
            get_crss = [real(kind=real64) :: 1.0, 2.0, 3.0, inf]
        else if (fabric == 1) then  ! B-type
            get_crss = [real(kind=real64) :: 3.0, 2.0, 1.0, inf]
        else if (fabric == 2) then  ! C-type
            get_crss = [real(kind=real64) :: 3.0, 2.0, inf, 1.0]
        else if (fabric == 3) then  ! D-type
            get_crss = [real(kind=real64) :: 1.0, 1.0, 3.0, inf]
        else if (fabric == 4) then  ! E-type
            get_crss = [real(kind=real64) :: 3.0, 1.0, 2.0, inf]
        end if
    else if (phase == 1) then  ! enstatite
        get_crss = [real(kind=real64) :: inf, inf, inf, 1.0]
        ! TODO: Throw error if fabric not equal to 5.
    end if
    ! TODO: Else throw error.
end function get_crss

pure function get_slip_invariants(strain_rate, orientation)
    ! Calculate strain rate invariants for minerals with four slip systems.
    real(kind=real64), intent(in), dimension(3,3) :: strain_rate, orientation
    real(kind=real64), dimension(4) :: get_slip_invariants

    integer :: i, j
    do i = 1, 3
        do j = 1, 3
            get_slip_invariants(1) = get_slip_invariants(1) &
                + strain_rate(i, j) * orientation(1, i) + orientation(2, j)
            get_slip_invariants(2) = get_slip_invariants(2) &
                + strain_rate(i, j) * orientation(1, i) + orientation(3, j)
            get_slip_invariants(3) = get_slip_invariants(3) &
                + strain_rate(i, j) * orientation(3, i) + orientation(2, j)
            get_slip_invariants(4) = get_slip_invariants(4) &
                + strain_rate(i, j) * orientation(3, i) + orientation(1, j)
        end do
    end do
end function get_slip_invariants

pure function get_slip_rates_olivine(invariants, slip_indices, ol_crss, deformation_exponent)
    ! Calculate relative slip rates of the active slip systems for olivine.
    real(kind=real64), intent(in), dimension(4) :: invariants, ol_crss
    integer, intent(in), dimension(4) :: slip_indices
    real(kind=real64), intent(in) :: deformation_exponent

    real(kind=real64), dimension(4) :: get_slip_rates_olivine

    real(kind=real64) :: prefactor, ratio_min, ratio_int
    integer :: i_inac, i_min, i_int, i_max
    i_inac = slip_indices(1)
    i_min = slip_indices(2)
    i_int = slip_indices(3)
    i_max = slip_indices(4)

    prefactor = ol_crss(i_max) / invariants(i_max)
    ratio_min = prefactor * invariants(i_min) / ol_crss(i_min)
    ratio_int = prefactor * invariants(i_int) / ol_crss(i_int)
    get_slip_rates_olivine(i_inac) = 0  ! Hardest system is completely inactive in olivine.
    get_slip_rates_olivine(i_min) = ratio_min * abs(ratio_min) ** (deformation_exponent - 1)
    get_slip_rates_olivine(i_int) = ratio_int * abs(ratio_int) ** (deformation_exponent - 1)
    get_slip_rates_olivine(i_max) = 1
end function get_slip_rates_olivine

pure function get_deformation_rate(phase, orientation, slip_rates)
    ! Calculate deformation rate tensor for olivine or enstatite.
    integer, intent(in) :: phase
    real(kind=real64), intent(in), dimension(3,3) :: orientation
    real(kind=real64), intent(in), dimension(4) :: slip_rates

    real(kind=real64), dimension(3,3) :: get_deformation_rate

    integer :: i, j
    do i = 1, 3
        do j = 1, 3
            get_deformation_rate(i, j) = 2 * ( &
                slip_rates(1) * orientation(1, i) * orientation(2, j) &
                + slip_rates(2) * orientation(1, i) * orientation(3, j) &
                + slip_rates(3) * orientation(3, i) * orientation(2, j) &
                + slip_rates(4) * orientation(3, i) * orientation(1, j) &
            )
        end do
    end do
end function get_deformation_rate

pure function get_slip_rate_softest(deformation_rate, velocity_gradient)
    ! Calculate dimensionless strain rate on the softest slip system.
    real(kind=real64), intent(in), dimension(3,3) :: deformation_rate, velocity_gradient
    real(kind=real64) :: get_slip_rate_softest

    real(kind=real64) :: enumerator, denominator

    integer :: j, k, L
    do j = 1, 3
        k = mod((j + 1), 3)  ! Using correction from Fraters & Billen 2021 S1.
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
    if (-1e-15 < denominator .and. denominator < 1e-15) then
        get_slip_rate_softest = 0e0
    end if
    get_slip_rate_softest = enumerator / denominator
end function get_slip_rate_softest

pure function get_orientation_change( &
    orientation, &
    velocity_gradient, &
    deformation_rate, &
    softest_rate, &
    permutation_symbol &
)
    ! Calculate the rotation for a grain undergoing dislocation creep.
    real(kind=real64), intent(in), dimension(3,3) :: orientation
    real(kind=real64), intent(in), dimension(3,3) :: deformation_rate, velocity_gradient
    real(kind=real64), intent(in), dimension(3,3,3) :: permutation_symbol
    real(kind=real64), intent(in) :: softest_rate

    real(kind=real64), dimension(3, 3) :: get_orientation_change

    real(kind=real64), dimension(3) :: spin_vector

    integer :: j, p, q, r, s
    do j = 1, 3
        r = mod((j + 1), 3)
        s = mod((j + 2), 3)
        spin_vector(j) = ((velocity_gradient(s, r) - velocity_gradient(r, s)) &
            - (deformation_rate(s, r) - deformation_rate(r, s)) * softest_rate) / 2
    end do

    do p = 1, 3
        do q = 1, 3
            do r = 1, 3
                do s = 1, 3
                    get_orientation_change(p, q) = get_orientation_change(p, q) &
                        + permutation_symbol(q, r, s) * orientation(p, s) * spin_vector(r)
                end do
            end do
        end do
    end do
end function get_orientation_change

pure function get_strain_energy( &
    ol_crss, &
    slip_rates, &
    slip_indices, &
    softest_rate, &
    stress_exponent, &
    deformation_exponent, &
    nucleation_efficiency &
)
    ! Calculate strain energy due to dislocations for an olivine grain.
    real(kind=real64), intent(in), dimension(4) :: ol_crss, slip_rates
    integer, intent(in), dimension(4) :: slip_indices
    real(kind=real64), intent(in) :: softest_rate, nucleation_efficiency
    real(kind=real64), intent(in) :: stress_exponent, deformation_exponent

    real(kind=real64) :: get_strain_energy

    ! Corrected for spurrious division by strain rate scale in Eq. 11, Kaminski 2004.
    integer :: i
    real(kind=real64) :: dislocation_density
    get_strain_energy = 0.0
    do i = 1, 3
        dislocation_density = (1 / ol_crss(i)) ** (deformation_exponent - stress_exponent) &
            * abs(slip_rates(i) * softest_rate) &
            ** (stress_exponent / deformation_exponent)
        ! Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
        get_strain_energy = get_strain_energy + dislocation_density &
            * exp(-nucleation_efficiency * dislocation_density**2)
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
    out_energy &
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

    real(kind=real64), dimension(4) :: crss, slip_invariants, slip_rates, array
    integer, dimension(4) :: slip_indices
    real(kind=real64), dimension(3,3) :: deformation_rate
    real(kind=real64) :: slip_rate_softest

    integer :: i, len
    crss = get_crss(phase, fabric)
    slip_invariants = get_slip_invariants(strain_rate, orientation)
    len = size(slip_invariants)
    do i = 1, 4
        array(i) = abs(slip_invariants(i) / crss(i))
    end do
    if (phase == 0) then  ! olivine
        slip_indices = argsort(array)
        slip_rates = get_slip_rates_olivine( &
            slip_invariants, slip_indices, crss, deformation_exponent &
        )
    else if (phase == 1) then  ! enstatite
        do i = 1, 4
            array(i) = 1 / crss(i)
        end do
        slip_indices = argsort(array)
        slip_rates = [0.0, 0.0, 0.0, 0.0]
        if (abs(slip_invariants(len)) > 1e-15) then
            slip_rates(len) = 1.0
        end if
    end if

    deformation_rate = get_deformation_rate(phase, orientation, slip_rates)
    slip_rate_softest = get_slip_rate_softest(deformation_rate, velocity_gradient)
    out_spin = get_orientation_change( &
        orientation, &
        velocity_gradient, &
        deformation_rate, &
        slip_rate_softest, &
        permutation_symbol &
    )
    out_energy = get_strain_energy( &
        crss, &
        slip_rates, &
        slip_indices, &
        slip_rate_softest, &
        stress_exponent, &
        deformation_exponent, &
        nucleation_efficiency &
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
    out_growth &
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

    real(kind=real64), dimension(n_grains) :: energies
    real(kind=real64), dimension(3,3) :: orientation_change
    real(kind=real64) :: energy, mean_energy

    integer :: grain_index
    if (regime == 4) then  ! matrix dislocation creep
        do grain_index = 1, n_grains
            call get_rotation_and_strain( &
                phase, &
                fabric, &
                orientations(grain_index,:,:), &
                strain_rate, &
                velocity_gradient, &
                stress_exponent, &
                deformation_exponent, &
                nucleation_efficiency, &
                permutation_symbol, &
                orientation_change, &
                energy &
            )
            out_spin(grain_index,:,:) = orientation_change
            energies(grain_index) = fractions(grain_index) * energy
        end do

        mean_energy = sum(energies)
        do grain_index = 1, n_grains
            out_growth(grain_index) = volume_fraction * gbm_mobility &
                * fractions(grain_index) * mean_energy - energies(grain_index)
        end do
    end if
    ! TODO: Else throw error?
end subroutine derivatives

end module core
