!!! DREX forward advected, for pure A-type olvine only.
!!! Parameter values and initial conditions are set in the subroutine `init0`.

module comvar

    double precision, dimension(3, 3) :: Lij  ! Velocity gradient
    double precision, dimension(3, 3) :: Dij  ! Deformation rate
    double precision, dimension(3, 3) :: Fij  ! deformation gradient
    integer :: n_steps ! number of timesteps (strain-steps)
    double precision :: t_loc  ! time at each step
    double precision :: epsnot ! reference strain rate
    integer :: size3, size ! size = size3^3, number of grains
    double precision :: lambda, Mob, chi
    ! lambda = nucleation parameter
    ! Mob = grain mobility
    ! chi = threshold volume fraction for activation of grain boundary sliding

    double precision, dimension(3, 3, 3) :: alt ! \epsilon_{ijk} tensor
    double precision, dimension(3, 3) :: del ! \delta_{ij} tensor
    integer, dimension(3, 3) :: ijkl ! tensor of indices to form Cijkl from Sij
    integer, dimension(6) :: l1, l2 ! tensot of indices to form Sij from Cijkl

    double precision :: stressexp ! stress exponent
    double precision, dimension(4) :: tau
    ! RSS for the 4 slip systems of olivine (only 3 are activated)

    double precision, dimension(:), allocatable :: odf, dotodf, odfi ! grain volume fractions
    double precision, dimension(:), allocatable :: rt ! dislocation density
    double precision, dimension(:, :, :), allocatable :: acs, dotacs, acsi, acs0
    ! direction cosine matrices

    double precision, dimension(:), allocatable :: phi_fse  ! angle of long-axis of FSE
    double precision, dimension(:), allocatable :: ln_fse ! ln(longaxis/shortaxis) of FSE
    double precision, dimension(:), allocatable :: phi_a ! average olivine a-axis
    double precision, dimension(:), allocatable :: perc_a ! % of of S wave anisotropy

    double precision, dimension(6, 6) :: S0, Sav ! stiffness matrix
    double precision, dimension(3, 3, 3, 3) :: Cav ! Cijkl tensor at point of calculation

end module comvar


program DREX
    use comvar
    implicit none

    integer :: t  ! timestep loop variable
    double precision, dimension(3, 3) :: Bij ! left-strech tensor for FSE calculation
    integer :: nrot ! number of rotations for the Jacobi
    double precision, dimension(3) :: evals
    double precision, dimension(3, 3) :: evects
    ! eigen values and vectors in jacobi

    call init0
    ! Mob, chi, stressexp, lambda, size, Lij, Dij, Fij, alt, del, S0, acs0, ijkl, l1, l2
    phi_fse = 0d0; ln_fse = 0d0; phi_a = 0d0; perc_a = 0d0
    rt = 0d0; odf = 1d0/real(size3**3); acs = acs0
    ! Set up initial values and parameters, and intialise outputs

    do t = 1, n_steps
        call update_orientations(Fij)
        ! Update orientations and gran volume fractions

        Bij = matmul(Fij, transpose(Fij))
        ! Left-stretch tensor for FSE calculation, B = FF^T

        call JACOBI(Bij, 3, 3, evals, evects, nrot)
        ! Calculate the orientation of the long axis of the FSE

        if (evals(1) .eq. maxval(evals)) then
            phi_fse(t) = atan2(evects(3, 1), evects(1, 1))
        else if (evals(2) .eq. maxval(evals)) then
            phi_fse(t) = atan2(evects(3, 2), evects(1, 2))
        else
            phi_fse(t) = atan2(evects(3, 3), evects(1, 3))
        end if

        ln_fse(t) = 0.5*log(maxval(evals)/minval(evals))
        ! natural strain = ln(a/c) where a is the long axis = maxval(evals)**0.5

        call voigt
        ! Cijkl tensor (using Voigt average)

        call DECSYM(Sav, perc_a(t), phi_a(t))
        ! Percentage of anisotropy and orientation of axis of hexagonal symmetry

    end do
    write(6, *) phi_a

end program DREX


subroutine show(Aij, d)  ! Pretty print square matrix in C order.
    integer :: d
    double precision, dimension(d, d) :: Aij
    do i = 1,ubound(Aij,1)
        write(6, *) Aij(i,:)
    end do
end subroutine show


subroutine init0
    use comvar
    implicit none

    integer :: i, j1, j2, j3 ! loop counters
    double precision, dimension(:), allocatable :: ran0
    ! vector of random numbers used to generate initial random CPO

    double precision :: phi1, theta, phi2  ! eulerian angles
    double precision, dimension(:), allocatable :: xe1, xe2, xe3
    ! vectors of initial random eulerian angles

    ! double precision :: PI
    ! PI = acos(-1d0)
    ! Used for setting initial orientations if not random.

    Mob = 0d0  ! grain boundary mobility
    chi = 0d0  ! threshold for grain boundary sliding
    stressexp = 3.5d0 ! stress exponent
    lambda = 5d0 ! nucleation efficiency parameter
    size3 = 15
    size = size3**3
    ! number of (unrecrystallized, initial) grains, really just number of samples in SO(3)

    n_steps = 200
    t_loc = 0.005d0

    tau = reshape((/ 1d0, 2d0, 3d0, 1d60 /), shape(tau))  ! CRSS

    Lij = reshape((/ 0, 0, 0, 0, 0, 0, 2, 0, 0 /), shape(Lij))  ! dv_x/dz is nonzero, simple shear
    Dij = reshape((/ 0, 0, 1, 0, 0, 0, 1, 0, 0 /), shape(Dij))  ! D = 1/2 L + L^T
    epsnot = 1d0

    Fij = reshape((/ 1, 0, 0, 0, 1, 0, 0, 0, 1 /), shape(Fij))  ! identity, undeformed initial state

    alt = 0d0
    alt(1, 2, 3) = 1d0; alt(2, 3, 1) = 1d0; alt(3, 1, 2) = 1d0
    alt(1, 3, 2) = -1d0; alt(2, 1, 3) = -1d0; alt(3, 2, 1) = -1d0
    ! tensor \epsilon_{ijk}

    del = 0d0
    del(1, 1) = 1d0; del(2, 2) = 1d0; del(3, 3) = 1d0
    ! tensor \delta_{ij}

    ijkl(1, 1) = 1; ijkl(1, 2) = 6; ijkl(1, 3) = 5
    ijkl(2, 1) = 6; ijkl(2, 2) = 2; ijkl(2, 3) = 4
    ijkl(3, 1) = 5; ijkl(3, 2) = 4; ijkl(3, 3) = 3

    l1(1) = 1; l1(2) = 2; l1(3) = 3
    l1(4) = 2; l1(5) = 3; l1(6) = 1
    l2(1) = 1; l2(2) = 2; l2(3) = 3
    l2(4) = 3; l2(5) = 1; l2(6) = 2
    ! tensors of indices

    S0(1, 1) = 320.71d0; S0(1, 2) = 69.84d0; S0(1, 3) = 71.22d0
    S0(2, 1) = S0(1, 2); S0(2, 2) = 197.25d0; S0(2, 3) = 74.8d0
    S0(3, 1) = S0(1, 3); S0(3, 2) = S0(2, 3); S0(3, 3) = 234.32d0
    S0(4, 4) = 63.77d0; S0(5, 5) = 77.67d0; S0(6, 6) = 78.36d0
    ! Stiffness matrix for Olivine (GigaPascals)

    allocate (xe1(size), xe2(size), xe3(size))
    allocate (odf(size), odfi(size), dotodf(size))
    allocate (rt(size))
    allocate (ran0(3*size))
    allocate (acs(size, 3, 3), dotacs(size, 3, 3), acsi(size, 3, 3), acs0(size, 3, 3))
    allocate (phi_fse(n_steps), ln_fse(n_steps), phi_a(n_steps), perc_a(n_steps))

    call random_number(ran0)
    i = 1
    do j1 = 1, size3; do j2 = 1, size3; do j3 = 1, size3
        xe1(i) = (real(j1) - ran0(i))/real(size3)*acos(-1d0)
        xe2(i) = acos(-1d0 + (real(j2) - ran0(size + i))/real(size3)*2d0)
        xe3(i) = (real(j3) - ran0(i + 2*size))/real(size3)*acos(-1d0)
        i = i + 1
    end do; end do; end do
    ! initialization of orientations - uniformally random distribution
    ! Rmq cos(theta) used to sample the metric Eulerian space

    ! xe1 = - PI / 2
    ! xe2 = PI / 4
    ! xe3 = PI / 2
    ! Start all at 45 instead.

    do i = 1, size
        phi1 = xe1(i); theta = xe2(i); phi2 = xe3(i)

        ! Direction cosine matrix

        acs0(i, 1, 1) = cos(phi2)*cos(phi1) - cos(theta)*sin(phi1)*sin(phi2)
        acs0(i, 1, 2) = cos(phi2)*sin(phi1) + cos(theta)*cos(phi1)*sin(phi2)
        acs0(i, 1, 3) = sin(phi2)*sin(theta)

        acs0(i, 2, 1) = -sin(phi2)*cos(phi1) - cos(theta)*sin(phi1)*cos(phi2)
        acs0(i, 2, 2) = -sin(phi2)*sin(phi1) + cos(theta)*cos(phi1)*cos(phi2)
        acs0(i, 2, 3) = cos(phi2)*sin(theta)

        acs0(i, 3, 1) = sin(theta)*sin(phi1)
        acs0(i, 3, 2) = -sin(theta)*cos(phi1)
        acs0(i, 3, 3) = cos(theta)

    end do

    deallocate (xe1, xe2, xe3, ran0)

    return

1000 format(16(1pe14.6))

end subroutine init0


! Formerly subroutine strain, name change for searchability
subroutine update_orientations(fse)
    use comvar
    implicit none

    integer :: j, j1, j2, nn, n_iter ! loop counters
    double precision :: dt ! RK45 time step
    double precision, dimension(3, 3) :: fse, fsei ! local finite deformation tensor
    double precision, dimension(3, 3) :: kfse1, kfse2, kfse3, kfse4
    double precision, dimension(size) :: kodf1, kodf2, kodf3, kodf4
    double precision, dimension(size, 3, 3) :: kac1, kac2, kac3, kac4
    ! Arrays for storing RK45 intermediate values

    ! dt = min(t_loc, 1d-2/epsnot)  ! RK45 timestep
    dt = t_loc
    ! n_iter = nint(t_loc/dt)  ! number of iterations in the LPO loop
    n_iter = 1

    do nn = 1, n_iter

        fsei = fse
        odfi = odf; acsi = acs

        call deriv

        kfse1 = matmul(Lij, fsei)*dt
        kodf1 = dotodf*dt*epsnot
        kac1 = dotacs*dt*epsnot

        fsei = fse + 0.5d0*kfse1
        odfi = odf + 0.5d0*kodf1
        acsi = acs + 0.5d0*kac1

        do j = 1, size; do j1 = 1, 3; do j2 = 1, 3
            if (acsi(j, j1, j2) .gt. 1d0) acsi(j, j1, j2) = 1d0
            if (acsi(j, j1, j2) .lt. -1d0) acsi(j, j1, j2) = -1d0
        end do; end do; end do
        do j = 1, size
        if (odfi(j) .le. 0) odfi(j) = 0d0
        end do
        odfi = odfi/sum(odfi)

        call deriv

        kfse2 = matmul(Lij, fsei)*dt
        kodf2 = dotodf*dt*epsnot
        kac2 = dotacs*dt*epsnot

        fsei = fse + 0.5d0*kfse2
        odfi = odf + 0.5d0*kodf2
        acsi = acs + 0.5d0*kac2

        do j = 1, size; do j1 = 1, 3; do j2 = 1, 3
            if (acsi(j, j1, j2) .gt. 1d0) acsi(j, j1, j2) = 1d0
            if (acsi(j, j1, j2) .lt. -1d0) acsi(j, j1, j2) = -1d0
        end do; end do; end do
        do j = 1, size
        if (odfi(j) .le. 0) odfi(j) = 0d0
        end do
        odfi = odfi/sum(odfi)

        call deriv

        kfse3 = matmul(Lij, fsei)*dt
        kodf3 = dotodf*dt*epsnot
        kac3 = dotacs*dt*epsnot

        fsei = fse + kfse3
        odfi = odf + kodf3
        acsi = acs + kac3

        do j = 1, size; do j1 = 1, 3; do j2 = 1, 3
            if (acsi(j, j1, j2) .gt. 1d0) acsi(j, j1, j2) = 1d0
            if (acsi(j, j1, j2) .lt. -1d0) acsi(j, j1, j2) = -1d0
        end do; end do; end do
        do j = 1, size
        if (odfi(j) .le. 0) odfi(j) = 0d0
        end do
        odfi = odfi/sum(odfi)

        call deriv

        kfse4 = matmul(Lij, fsei)*dt
        kodf4 = dotodf*dt*epsnot
        kac4 = dotacs*dt*epsnot

        fse = fse + (kfse1/2d0 + kfse2 + kfse3 + kfse4/2d0)/3d0
        acs = acs + (kac1/2d0 + kac2 + kac3 + kac4/2d0)/3d0
        odf = odf + (kodf1/2d0 + kodf2 + kodf3 + kodf4/2d0)/3d0

        do j = 1, size; do j1 = 1, 3; do j2 = 1, 3
            if (acs(j, j1, j2) .gt. 1d0) acs(j, j1, j2) = 1d0
            if (acs(j, j1, j2) .lt. -1d0) acs(j, j1, j2) = -1d0
        end do; end do; end do
        ! do j = 1, size
        ! if (odfi(j) .le. 0) odfi(j) = 0d0
        ! end do

        odf = odf/sum(odf)

    end do

    return

end subroutine update_orientations


subroutine deriv
    use comvar
    implicit none

    integer :: i, i1, i2, i3, i4, j, k ! counters
    integer :: imax, iint, imin, iinac ! dummies
    integer, dimension(1) :: ti ! reordering array

    double precision :: Emean, rt1, rt2, rt3
    ! surface averaged aggregate NRJ
    ! dislocation density for each slip system

    double precision :: gam0
    ! slip rate on the softest slip system

    double precision :: R1, R2
    double precision :: qint, qmin, sn1, rat
    ! dummies

    double precision, dimension(4) :: bigi, q, qab ! intermediates for G calc

    double precision, dimension(4) :: gam
    ! ratios of strain between softest slip system and slip system s for Olivine

    double precision, dimension(3) :: rot
    ! rotation rate vector

    double precision, dimension(3, 3) :: g
    ! slip tensor

    double precision, dimension(3, 3) :: lx, ex
    ! dimensionless velocity gradient and strain rate tensors

    lx = Lij/epsnot; ex = Dij/epsnot
    ! Dimensionless strain rate and velocity gradient tensors

    do i = 1, size
        bigi = 0d0; gam = 0d0; g = 0d0

        do i1 = 1, 3; do i2 = 1, 3
            bigi(1) = bigi(1) + ex(i1, i2)*acsi(i, 1, i1)*acsi(i, 2, i2)
            bigi(2) = bigi(2) + ex(i1, i2)*acsi(i, 1, i1)*acsi(i, 3, i2)
            bigi(3) = bigi(3) + ex(i1, i2)*acsi(i, 3, i1)*acsi(i, 2, i2)
            bigi(4) = bigi(4) + ex(i1, i2)*acsi(i, 3, i1)*acsi(i, 1, i2)
        end do; end do
        ! Calculate invariants e_{pr} T_{pr} for the four slip systems of olivine

        q = bigi/tau
        ! Quotients I/tau

        qab = abs(q)
        ti = maxloc(qab); imax = ti(1); qab(imax) = -1d0
        ti = maxloc(qab); iint = ti(1); qab(iint) = -1d0
        ti = maxloc(qab); imin = ti(1); qab(imin) = -1d0
        ti = maxloc(qab); iinac = ti(1)
        ! Reorder quotients I/tau according to absolute magnitude

        gam(imax) = 1d0
        ! Calculate weighting factors gam_s relative to value gam_i for which
        ! I/tau is largest

        rat = tau(imax)/bigi(imax)
        qint = rat*bigi(iint)/tau(iint)
        qmin = rat*bigi(imin)/tau(imin)
        sn1 = stressexp - 1d0

        gam(iint) = qint*(abs(qint))**sn1
        gam(imin) = qmin*(abs(qmin))**sn1
        gam(iinac) = 0d0

        do i1 = 1, 3; do i2 = 1, 3
            g(i1, i2) = 2d0*(gam(1)*acsi(i, 1, i1)*acsi(i, 2, i2) + &
                            gam(2)*acsi(i, 1, i1)*acsi(i, 3, i2) + &
                            gam(3)*acsi(i, 3, i1)*acsi(i, 2, i2) + &
                            gam(4)*acsi(i, 3, i1)*acsi(i, 1, i2))
        end do; end do
        ! calculation of G tensor

        R1 = 0d0; R2 = 0d0
        do j = 1, 3
            i2 = j + 2
            if (i2 .gt. 3) i2 = i2 - 3

            R1 = R1 - (g(j, i2) - g(i2, j))*(g(j, i2) - g(i2, j))
            R2 = R2 - (g(j, i2) - g(i2, j))*(lx(j, i2) - lx(i2, j))

            do k = 1, 3

                R1 = R1 + 2d0*g(j, k)*g(j, k)
                R2 = R2 + 2d0*lx(j, k)*g(j, k)

            end do
        end do
        gam0 = R2/R1
        ! calculation of strain rate on the softest slip system

        rt1 = tau(1)**(1.5d0 - stressexp)*abs(gam(1)*gam0)**(1.5d0/stressexp)
        rt2 = tau(2)**(1.5d0 - stressexp)*abs(gam(2)*gam0)**(1.5d0/stressexp)
        rt3 = tau(3)**(1.5d0 - stressexp)*abs(gam(3)*gam0)**(1.5d0/stressexp)
        rt(i) = rt1*exp(-lambda*rt1**2) + &
                rt2*exp(-lambda*rt2**2) + &
                rt3*exp(-lambda*rt3**2)
        ! dislocation density calculation

        rot(3) = (lx(2, 1) - lx(1, 2))/2d0 - (g(2, 1) - g(1, 2))/2d0*gam0
        rot(2) = (lx(1, 3) - lx(3, 1))/2d0 - (g(1, 3) - g(3, 1))/2d0*gam0
        rot(1) = (lx(3, 2) - lx(2, 3))/2d0 - (g(3, 2) - g(2, 3))/2d0*gam0
        ! calculation of the rotation rate

        dotacs(i, :, :) = 0d0
        do i1 = 1, 3; do i2 = 1, 3; do i3 = 1, 3; do i4 = 1, 3
            dotacs(i, i1, i2) = dotacs(i, i1, i2) + alt(i2, i3, i4)*acsi(i, i1, i4)*rot(i3)
        end do; end do; end do; end do
        ! derivative of the matrix of direction cosine

        ! grain boundary sliding for small grains
        if (odfi(i) .lt. chi/real(size)) then
            dotacs(i, :, :) = 0d0
            rt(i) = 0d0
        end if

    end do

    Emean = sum(odfi*rt)
    ! Volume averaged energy
    do i = 1, size
        dotodf(i) = Mob*odfi(i)*(Emean - rt(i))
    end do
    ! Change of volume fraction by grain boundary migration

    return

end subroutine deriv


subroutine voigt
    use comvar
    implicit none

    integer :: i, j, k, ll, n, nu, p, q, r, ss

    double precision, dimension(3, 3, 3, 3) :: C0, Cav2

    C0 = 0d0; Cav = 0d0; Sav = 0d0

    ! Single-xl elastic tensors c0_{ijkl}
    do i = 1, 3; do j = 1, 3; do k = 1, 3; do ll = 1, 3
        C0(i, j, k, ll) = S0(ijkl(i, j), ijkl(k, ll))
    end do; end do; end do; end do

    do nu = 1, size
        Cav2 = 0d0
        do i = 1, 3; do j = 1, 3; do k = 1, 3; do ll = 1, 3
            do p = 1, 3; do q = 1, 3; do r = 1, 3; do ss = 1, 3
                    Cav2(i, j, k, ll) = Cav2(i, j, k, ll) + &
                                        acs(nu, p, i)*acs(nu, q, j)*acs(nu, r, k)*acs(nu, ss, ll)*C0(p, q, r, ss)
            end do; end do; end do; end do
            Cav(i, j, k, ll) = Cav(i, j, k, ll) + Cav2(i, j, k, ll)*odf(nu)
        end do; end do; end do; end do
    end do

    do i = 1, 6; do j = 1, 6
        Sav(i, j) = Cav(l1(i), l2(i), l1(j), l2(j))
    end do; end do
    ! Average stiffness matrix

    return

end subroutine voigt


module DECMOD

   double precision, dimension(3, 3) :: SCC
   double precision, dimension(6, 6) :: CE1
   double precision, dimension(3, 3, 3, 3) :: EL1
   double precision, dimension(21) :: XEC
   double precision :: XN, ANIS

end module DECMOD

!
!****************************************************************
!

subroutine DECSYM(CED, PERC, INCLTI)

   use DECMOD

   implicit none

   double precision, dimension(6, 6) :: CED
   double precision :: PERC, INCLTI, DC5, PI
   double precision, dimension(3) :: TIAXIS

   PI = acos(-1d0)
   CE1 = CED
   EL1 = 0d0
   call FULLSYM6(CE1)
   call TENS4(CE1, EL1)
   call V21D(CE1, XEC)
   XN = sqrt(dot_product(XEC, XEC))

   call SCCA
   call PROJECTI(XEC, DC5)

   PERC = (ANIS - DC5)/XN*100d0

   TIAXIS = SCC(3, :)
   TIAXIS = TIAXIS/sqrt(sum(TIAXIS*TIAXIS))
   INCLTI = asin(TIAXIS(3))

   return

end subroutine DECSYM

!
!****************************************************************
!

subroutine FULLSYM6(C)

   implicit none

   double precision, dimension(6, 6) :: C

   C(3, 2) = C(2, 3)
   C(3, 1) = C(1, 3)
   C(2, 1) = C(1, 2)

   C(4, 1) = C(1, 4)
   C(5, 1) = C(1, 5)
   C(6, 1) = C(1, 6)
   C(4, 2) = C(2, 4)
   C(5, 2) = C(2, 5)
   C(6, 2) = C(2, 6)
   C(4, 3) = C(3, 4)
   C(5, 3) = C(3, 5)
   C(6, 3) = C(3, 6)

   C(6, 5) = C(5, 6)
   C(6, 4) = C(4, 6)
   C(5, 4) = C(4, 5)

   return

end subroutine FULLSYM6

!
!****************************************************************
!

subroutine TENS4(C, C4)

   implicit none

   integer :: i, j, k, l
   integer :: p, q
   integer :: NDELTA
   double precision, dimension(6, 6) :: C
   double precision, dimension(3, 3, 3, 3) :: C4

   C4 = 0d0

   do i = 1, 3
   do j = 1, 3
   do k = 1, 3
   do l = 1, 3

      p = NDELTA(i, j)*i + (1 - NDELTA(i, j))*(9 - i - j)
      q = NDELTA(k, l)*k + (1 - NDELTA(k, l))*(9 - k - l)
      C4(i, j, k, l) = C(p, q)

   end do
   end do
   end do
   end do

end subroutine TENS4

!
!****************************************************************
!

function NDELTA(i, j)

   implicit none

   integer :: i, j
   integer :: NDELTA

   NDELTA = 0
   if (i == j) NDELTA = 1

end function NDELTA

!
!****************************************************************
!

subroutine JACOBI(a, n, np, d, v, nrot)

   ! Jacobi algorithm for real symmetric matrix
   ! Gives eigenvalues and orthonormalized eigenvectors
   ! Half of the input matrix is crushed

   implicit none

   integer :: n, np, nrot
   integer, parameter :: NMAX = 500, IDP = kind(1d0)

   double precision, dimension(np, np) :: a, v
   double precision, dimension(np) :: d
   double precision, dimension(NMAX) :: b, z

   integer :: i, ip, iq, j
   double precision :: c, g, h, s, sm, t, tau, theta, tresh

   do ip = 1, n
      do iq = 1, n
         v(ip, iq) = 0d0
      end do
      v(ip, ip) = 1d0
   end do
   do ip = 1, n
      b(ip) = a(ip, ip)
      d(ip) = b(ip)
      z(ip) = 0d0
   end do
   nrot = 0
   do i = 1, 50
      sm = 0d0
      do ip = 1, n - 1
      do iq = ip + 1, n
         sm = sm + abs(a(ip, iq))
      end do
      end do
      if (sm == 0d0) return
      if (i < 4) then
         tresh = 0.2d0*sm/real(n, IDP)**2d0
      else
         tresh = 0d0
      end if
      do ip = 1, n - 1
      do iq = ip + 1, n
         g = 100d0*abs(a(ip, iq))
         if ((i > 4) .and. (abs(d(ip)) + &
                            g == abs(d(ip))) .and. (abs(d(iq)) + g == abs(d(iq)))) then
            a(ip, iq) = 0d0
         else if (abs(a(ip, iq)) > tresh) then
            h = d(iq) - d(ip)
            if (abs(h) + g == abs(h)) then
               t = a(ip, iq)/h
            else
               theta = 0.5d0*h/a(ip, iq)
               t = 1d0/(abs(theta) + sqrt(1d0 + theta**2d0))
               if (theta < 0d0) t = -t
            end if
            c = 1d0/sqrt(1d0 + t**2d0)
            s = t*c
            tau = s/(1d0 + c)
            h = t*a(ip, iq)
            z(ip) = z(ip) - h
            z(iq) = z(iq) + h
            d(ip) = d(ip) - h
            d(iq) = d(iq) + h
            a(ip, iq) = 0d0
            do j = 1, ip - 1
               g = a(j, ip)
               h = a(j, iq)
               a(j, ip) = g - s*(h + g*tau)
               a(j, iq) = h + s*(g - h*tau)
            end do
            do j = ip + 1, iq - 1
               g = a(ip, j)
               h = a(j, iq)
               a(ip, j) = g - s*(h + g*tau)
               a(j, iq) = h + s*(g - h*tau)
            end do
            do j = iq + 1, n
               g = a(ip, j)
               h = a(iq, j)
               a(ip, j) = g - s*(h + g*tau)
               a(iq, j) = h + s*(g - h*tau)
            end do
            do j = 1, n
               g = v(j, ip)
               h = v(j, iq)
               v(j, ip) = g - s*(h + g*tau)
               v(j, iq) = h + s*(g - h*tau)
            end do
            nrot = nrot + 1
         end if
      end do
      end do
      do ip = 1, n
         b(ip) = b(ip) + z(ip)
         d(ip) = b(ip)
         z(ip) = 0d0
      end do
   end do
   write (6, '(''Too many iterations in JACOBI'')')

   return

end subroutine JACOBI

!
!****************************************************************
!

subroutine EIGSRT(d, v, n, np)

   ! Order eigenvalues and eigenvectors
   ! 1 : max
   ! 2 : mid
   ! 3 : min

   implicit none

   integer :: np, n
   integer :: i, j, k
   double precision, dimension(np) :: d
   double precision, dimension(np, np) :: v
   double precision :: p

   do i = 1, n - 1
      k = i
      p = d(i)
      do j = i + 1, n
         if (d(j) >= p) then
            k = j
            p = d(j)
         end if
      end do
      if (k /= i) then
         d(k) = d(i)
         d(i) = p
         do j = 1, n
            p = v(j, i)
            v(j, i) = v(j, k)
            v(j, k) = p
         end do
      end if
   end do

   return

end subroutine EIGSRT

!
!****************************************************************
!

subroutine MAT6(C4, C)

   implicit none

   integer :: i
   double precision, dimension(6, 6) :: C
   double precision, dimension(3, 3, 3, 3) :: C4

   C = 0d0

   do i = 1, 3
      C(i, i) = C4(i, i, i, i)
   end do
   do i = 2, 3
      C(1, i) = (C4(1, 1, i, i) + C4(i, i, 1, 1))/2d0
      C(i, 1) = C(1, i)
   end do
   C(2, 3) = (C4(2, 2, 3, 3) + C4(3, 3, 2, 2))/2d0
   C(3, 2) = C(2, 3)

   do i = 1, 3
      C(i, 4) = (C4(i, i, 2, 3) + C4(i, i, 3, 2) + &
                 C4(2, 3, i, i) + C4(3, 2, i, i))/4d0
      C(4, i) = C(i, 4)
   end do
   do i = 1, 3
      C(i, 5) = (C4(i, i, 1, 3) + C4(i, i, 3, 1) + &
                 C4(1, 3, i, i) + C4(3, 1, i, i))/4d0
      C(5, i) = C(i, 5)
   end do
   do i = 1, 3
      C(i, 6) = (C4(i, i, 1, 2) + C4(i, i, 2, 1) + &
                 C4(1, 2, i, i) + C4(2, 1, i, i))/4d0
      C(6, i) = C(i, 6)
   end do

   C(4, 4) = (C4(2, 3, 2, 3) + C4(2, 3, 3, 2) + &
              C4(3, 2, 2, 3) + C4(3, 2, 3, 2))/4d0
   C(5, 5) = (C4(1, 3, 1, 3) + C4(1, 3, 3, 1) + &
              C4(3, 1, 1, 3) + C4(3, 1, 3, 1))/4d0
   C(6, 6) = (C4(2, 1, 2, 1) + C4(2, 1, 1, 2) + &
              C4(1, 2, 2, 1) + C4(1, 2, 1, 2))/4d0
   C(4, 5) = (C4(2, 3, 1, 3) + C4(2, 3, 3, 1) + &
              C4(3, 2, 1, 3) + C4(3, 2, 3, 1) + &
              C4(1, 3, 2, 3) + C4(1, 3, 3, 2) + &
              C4(3, 1, 2, 3) + C4(3, 1, 3, 2))/8d0

   C(5, 4) = C(4, 5)
   C(4, 6) = (C4(2, 3, 1, 2) + C4(2, 3, 2, 1) + &
              C4(3, 2, 1, 2) + C4(3, 2, 2, 1) + &
              C4(1, 2, 2, 3) + C4(1, 2, 3, 2) + &
              C4(2, 1, 2, 3) + C4(2, 1, 3, 2))/8d0
   C(6, 4) = C(4, 6)
   C(5, 6) = (C4(1, 3, 1, 2) + C4(1, 3, 2, 1) + &
              C4(3, 1, 1, 2) + C4(3, 1, 2, 1) + &
              C4(1, 2, 1, 3) + C4(1, 2, 3, 1) + &
              C4(2, 1, 1, 3) + C4(2, 1, 3, 1))/8d0
   C(6, 5) = C(5, 6)

   return

end subroutine MAT6

!
!****************************************************************
!

subroutine PERMUT(INDEX, PERM)

   implicit none

   integer :: INDEX
   integer, dimension(3) :: PERM

   if (INDEX == 1) then
      PERM(1) = 1
      PERM(2) = 2
      PERM(3) = 3
   end if
   if (INDEX == 2) then
      PERM(1) = 2
      PERM(2) = 3
      PERM(3) = 1
   end if
   if (INDEX == 3) then
      PERM(1) = 3
      PERM(2) = 1
      PERM(3) = 2
   end if

   return

end subroutine PERMUT

!
!****************************************************************
!

subroutine ROT4(C4, R, C4C)

   implicit none

   integer :: i1, i2, i3, i4, j1, j2, j3, j4
   double precision, dimension(3, 3, 3, 3) :: C4, C4C
   double precision, dimension(3, 3) :: R

   C4C = 0d0

   do i1 = 1, 3
   do i2 = 1, 3
   do i3 = 1, 3
   do i4 = 1, 3

      do j1 = 1, 3
      do j2 = 1, 3
      do j3 = 1, 3
      do j4 = 1, 3

         C4C(i1, i2, i3, i4) = C4C(i1, i2, i3, i4) + &
                               R(i1, j1)*R(i2, j2)*R(i3, j3)*R(i4, j4)*C4(j1, j2, j3, j4)

      end do
      end do
      end do
      end do

   end do
   end do
   end do
   end do

   return

end subroutine ROT4

!
!****************************************************************
!

subroutine SCCA

   use DECMOD

   implicit none

   integer :: i, NROT, i1, i2, NDVC
   integer, dimension(3) :: IHS
   double precision, dimension(3) :: EGDI, EGVO
   double precision, dimension(3, 3) :: DI, VO, VECDI, VECVO
   double precision, dimension(6, 6) :: CEC
   double precision, dimension(3, 3, 3, 3) :: ELC
   double precision, dimension(21) :: XH, XD
   double precision :: SDV, ADV, ADVC, SCN, DEV, K, G

   DI = 0d0
   VO = 0d0
   K = 0d0
   G = 0d0

   do i = 1, 3
      DI(1, 1) = CE1(1, i) + DI(1, 1)
      DI(2, 2) = CE1(2, i) + DI(2, 2)
      DI(3, 3) = CE1(3, i) + DI(3, 3)
      DI(2, 1) = CE1(6, i) + DI(2, 1)
      DI(3, 1) = CE1(5, i) + DI(3, 1)
      DI(3, 2) = CE1(4, i) + DI(3, 2)
   end do
   DI(1, 2) = DI(2, 1)
   DI(1, 3) = DI(3, 1)
   DI(2, 3) = DI(3, 2)

   VO(1, 1) = CE1(1, 1) + CE1(6, 6) + CE1(5, 5)
   VO(2, 2) = CE1(6, 6) + CE1(2, 2) + CE1(4, 4)
   VO(3, 3) = CE1(5, 5) + CE1(4, 4) + CE1(3, 3)
   VO(2, 1) = CE1(1, 6) + CE1(2, 6) + CE1(4, 5)
   VO(1, 2) = VO(2, 1)
   VO(3, 1) = CE1(1, 5) + CE1(3, 5) + CE1(4, 6)
   VO(1, 3) = VO(3, 1)
   VO(3, 2) = CE1(2, 4) + CE1(3, 4) + CE1(5, 6)
   VO(2, 3) = VO(3, 2)

   do i = 1, 3
      K = K + DI(i, i)
      G = G + VO(i, i)
   end do
   K = K/9d0
   G = G/10d0 - 3d0*K/10d0

   ! Anisotropy

   ANIS = 0d0
   XH = 0d0
   XD = 0d0
   XH(1) = K + 4d0*G/3d0
   XH(2) = K + 4d0*G/3d0
   XH(3) = K + 4d0*G/3d0
   XH(4) = sqrt(2d0)*(K - 2d0*G/3d0)
   XH(5) = sqrt(2d0)*(K - 2d0*G/3d0)
   XH(6) = sqrt(2d0)*(K - 2d0*G/3d0)
   XH(7) = 2d0*G
   XH(8) = 2d0*G
   XH(9) = 2d0*G

   XD = XEC - XH
   ANIS = sqrt(dot_product(XD, XD))

   ! Dil. and Voigt axes

   call JACOBI(DI, 3, 3, EGDI, VECDI, NROT)
   call EIGSRT(EGDI, VECDI, 3, 3)
   call JACOBI(VO, 3, 3, EGVO, VECVO, NROT)
   call EIGSRT(EGVO, VECVO, 3, 3)

   ! Search for SCCA directions

   do i1 = 1, 3
      NDVC = 0
      ADVC = 10d0
      SCN = 0d0
      do i2 = 1, 3
         SDV = dot_product(VECDI(:, i1), VECVO(:, i2))
         if (abs(SDV) >= 1d0) SDV = sign(1d0, SDV)
         ADV = acos(SDV)
         if (SDV < 0d0) ADV = acos(-1d0) - ADV
         if (ADV < ADVC) then
            NDVC = sign(1d0, SDV)*i2
            ADVC = ADV
         end if
      end do

      VECDI(:, i1) = (VECDI(:, i1) + NDVC*VECVO(:, abs(NDVC)))/2d0
      SCN = sqrt(VECDI(1, i1)**2d0 + VECDI(2, i1)**2d0 + &
                 VECDI(3, i1)**2d0)
      VECDI(:, i1) = VECDI(:, i1)/SCN
   end do

   ! Higher symmetry axis

   SCC = transpose(VECDI)
   ELC = 0d0
   SDV = XN
   do i = 1, 3
      call PERMUT(i, IHS)
      do i1 = 1, 3
         VECDI(i1, :) = SCC(IHS(i1), :)
      end do
      call ROT4(EL1, VECDI, ELC)
      call MAT6(ELC, CEC)
      call V21D(CEC, XEC)
      call PROJECTI(XEC, DEV)
      if (DEV < SDV) then
         SDV = DEV
         NDVC = i
      end if
   end do

   VECDI = SCC
   call PERMUT(NDVC, IHS)
   do i1 = 1, 3
      SCC(i1, :) = VECDI(IHS(i1), :)
   end do

   ! Rotate in SCCA

   call ROT4(EL1, SCC, ELC)
   call MAT6(ELC, CEC)
   call V21D(CEC, XEC)

   return

end subroutine SCCA

!
!****************************************************************
!

subroutine V21D(C, X)

   implicit none

   double precision, dimension(6, 6) :: C
   double precision, dimension(21) :: X

   X = 0d0
   X(1) = C(1, 1)
   X(2) = C(2, 2)
   X(3) = C(3, 3)
   X(4) = sqrt(2d0)*C(2, 3)
   X(5) = sqrt(2d0)*C(1, 3)
   X(6) = sqrt(2d0)*C(1, 2)
   X(7) = 2d0*C(4, 4)
   X(8) = 2d0*C(5, 5)
   X(9) = 2d0*C(6, 6)
   X(10) = 2d0*C(1, 4)
   X(11) = 2d0*C(2, 5)
   X(12) = 2d0*C(3, 6)
   X(13) = 2d0*C(3, 4)
   X(14) = 2d0*C(1, 5)
   X(15) = 2d0*C(2, 6)
   X(16) = 2d0*C(2, 4)
   X(17) = 2d0*C(3, 5)
   X(18) = 2d0*C(1, 6)
   X(19) = 2d0*sqrt(2d0)*C(5, 6)
   X(20) = 2d0*sqrt(2d0)*C(4, 6)
   X(21) = 2d0*sqrt(2d0)*C(4, 5)

   return

end subroutine V21D

!
!***************************************************************
!

subroutine PROJECTI(X, DEV)

   implicit none

   double precision :: DEV
   double precision, dimension(21) :: X, XH, XD

   XH = 0d0
   XD = 0d0
   DEV = 0d0

   XH(1) = 3d0/8d0*(X(1) + X(2)) + X(6)/4d0/sqrt(2d0) + X(9)/4d0
   XH(2) = XH(1)
   XH(3) = X(3)
   XH(4) = (X(4) + X(5))/2d0
   XH(5) = XH(4)
   XH(6) = (X(1) + X(2))/4d0/sqrt(2d0) + 3d0/4d0*X(6) &
           - X(9)/2d0/sqrt(2d0)
   XH(7) = (X(7) + X(8))/2d0
   XH(8) = XH(7)
   XH(9) = (X(1) + X(2))/4d0 - X(6)/2d0/sqrt(2d0) + X(9)/2d0

   XD = X - XH
   DEV = sqrt(dot_product(XD, XD))

   return

end subroutine PROJECTI
