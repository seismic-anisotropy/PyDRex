# This file contains alternative implementations of some PyDRex functions
# Line 14: Original strain, to be used with the original pathline
# Line 120: strain modified to accommodate for the new pathline
# Line 240: strain using Scipy solve_ivp, requiring the new pathline
# Line 327: Original eigen, necessary for the original pipar
# Line 365: Original pipar
# Line 415: New pipar, mainly a re-write of isacalc
# Line 473: Original pathline
# Line 530: New pathline, using Scipy solve_ivp
# Line 577: DRex matching the original implementation of pathline and strain
# Line 609: DRex matching the new implementation of pathline and strain
# Original strain, to be used with the original pathline
def strain(step, stream_Lij, stream_e, stream_dt, stream_alpha, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        acs0 = dictGlobals['acs0']
        chi = dictGlobals['chi']
        size = dictGlobals['size']
    """
    fse = np.identity(3)
    odf = 1 / size * np.ones(size)
    odf_ens = 1 / size * np.ones(size)
    acs = np.copy(acs0)
    acs_ens = np.copy(acs0)
    # step-number of steps used to construct the streamline
    for currStep in range(step, -1, -1):
        # Lij at time from record
        L = np.ascontiguousarray(stream_Lij[:, :, currStep])
        t_loc = stream_dt[currStep]
        epsnot = stream_e[currStep]  # epsnot: reference strain rate
        alpha = stream_alpha[currStep]  # alpha at current location
        e = (L + np.transpose(L)) / 2
        # time stepping for LPO calculation
        dt = min(t_loc, 1e-2 / epsnot)
        # number of iterations in the LPO loop
        N_strain = int(round(t_loc / dt))
        # LPO loop on the point on the pathline
        for nn in range(N_strain):
            # CALL 1/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acs, acs_ens, fse, odf, odf_ens, alpha
            )
            kfse1 = np.dot(L, fse) * dt
            kodf1 = dotodf * dt * epsnot
            kodf1_ens = dotodf_ens * dt * epsnot
            kac1 = dotacs * dt * epsnot
            kac1_ens = dotacs_ens * dt * epsnot
            fsei = fse + 0.5 * kfse1
            acsi = np.clip(acs + 0.5 * kac1, -1, 1)
            acsi_ens = np.clip(acs_ens + 0.5 * kac1_ens, -1, 1)
            odfi = np.clip(odf + 0.5 * kodf1, 0, None)
            odfi /= np.sum(odfi)
            odfi_ens = np.clip(odf_ens + 0.5 * kodf1_ens, 0, None)
            odfi_ens /= np.sum(odfi_ens)
            # CALL 2/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse2 = np.dot(L, fsei) * dt
            kodf2 = dotodf * dt * epsnot
            kodf2_ens = dotodf_ens * dt * epsnot
            kac2 = dotacs * dt * epsnot
            kac2_ens = dotacs_ens * dt * epsnot
            fsei = fse + 0.5 * kfse2
            acsi = np.clip(acs + 0.5 * kac2, -1, 1)
            acsi_ens = np.clip(acs_ens + 0.5 * kac2_ens, -1, 1)
            odfi = np.clip(odf + 0.5 * kodf2, 0, None)
            odfi /= np.sum(odfi)
            odfi_ens = np.clip(odf_ens + 0.5 * kodf2_ens, 0, None)
            odfi_ens /= np.sum(odfi_ens)
            # CALL 3/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse3 = np.dot(L, fsei) * dt
            kodf3 = dotodf * dt * epsnot
            kodf3_ens = dotodf_ens * dt * epsnot
            kac3 = dotacs * dt * epsnot
            kac3_ens = dotacs_ens * dt * epsnot
            fsei = fse + kfse3
            acsi = np.clip(acs + kac3, -1, 1)
            acsi_ens = np.clip(acs_ens + kac3_ens, -1, 1)
            odfi = np.clip(odf + kodf3, 0, None)
            odfi /= np.sum(odfi)
            odfi_ens = np.clip(odf_ens + kodf3_ens, 0, None)
            odfi_ens /= np.sum(odfi_ens)
            # CALL 4/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse4 = np.dot(L, fsei) * dt
            kodf4 = dotodf * dt * epsnot
            kodf4_ens = dotodf_ens * dt * epsnot
            kac4 = dotacs * dt * epsnot
            kac4_ens = dotacs_ens * dt * epsnot
            fse += (kfse1 / 2 + kfse2 + kfse3 + kfse4 / 2) / 3
            acs = np.clip(acs + (kac1 / 2 + kac2 + kac3 + kac4 / 2) / 3, -1, 1)
            acs_ens = np.clip(
                acs_ens + (kac1_ens / 2 + kac2_ens + kac3_ens + kac4_ens / 2) / 3, -1, 1
            )
            odf += (kodf1 / 2 + kodf2 + kodf3 + kodf4 / 2) / 3
            odf_ens += (kodf1_ens / 2 + kodf2_ens + kodf3_ens + kodf4_ens / 2) / 3
            odf /= np.sum(odf)
            odf_ens /= np.sum(odf_ens)
            # Grain-boundary sliding
            mask = odf <= chi / size
            mask_ens = odf_ens <= chi / size
            acs[mask, :, :] = acs0[mask, :, :]
            acs_ens[mask_ens, :, :] = acs0[mask_ens, :, :]
            odf[mask] = chi / size
            odf_ens[mask_ens] = chi / size
            odf /= np.sum(odf)
            odf_ens /= np.sum(odf_ens)
    return fse, acs, acs_ens, odf, odf_ens


# strain modified to accommodate for the new pathline
def strain(pathTime, pathDense, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        acs0 = dictGlobals['acs0']
        chi = dictGlobals['chi']
        gridCoords = dictGlobals['gridCoords']
        iDefMech = dictGlobals['iDefMech']
        size = dictGlobals['size']
    """
    fse = np.identity(3)
    acs = acs0.copy()
    acs_ens = acs0.copy()
    odf = np.ones(size) / size
    odf_ens = np.ones(size) / size
    for time in reversed(pathTime):
        if isInside(pathDense(time), dictGlobals):
            currTime = time
            break
    while currTime < pathTime[0]:
        currPoint = pathDense(currTime)
        currVel = interpVel(currPoint, dictGlobals)
        L, epsnot = interpVelGrad(currPoint, dictGlobals)
        e = (L + L.transpose()) / 2
        alpha = iDefMech(*currPoint)
        indLeft = [
            np.searchsorted(gridCoords[x], currPoint[x]) for x in range(currPoint.size)
        ]
        gridStep = []
        for coord, ind in zip(gridCoords, indLeft):
            try:
                gridStep.append(coord[ind + 1] - coord[ind])
            except IndexError:
                gridStep.append(coord[ind] - coord[ind - 1])
        dtPathline = min(gridStep) / 4 / norm(currVel, ord=2)
        dtPathline = min(dtPathline, pathTime[0] - currTime)
        # time stepping for LPO calculation
        dt = min(dtPathline, 1e-2 / epsnot)
        # number of iterations in the LPO loop
        nbIter = int(dtPathline / dt)
        # LPO loop on the point on the pathline
        for iter in range(nbIter):
            # CALL 1/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acs, acs_ens, fse, odf, odf_ens, alpha
            )
            kfse1 = np.dot(L, fse) * dt
            kodf1 = dotodf * dt * epsnot
            kodf1_ens = dotodf_ens * dt * epsnot
            kac1 = dotacs * dt * epsnot
            kac1_ens = dotacs_ens * dt * epsnot
            fsei = fse + 0.5 * kfse1
            acsi = np.clip(acs + 0.5 * kac1, -1, 1)
            acsi_ens = np.clip(acs_ens + 0.5 * kac1_ens, -1, 1)
            odfi = np.clip(odf + 0.5 * kodf1, 0, None)
            odfi /= odfi.sum()
            odfi_ens = np.clip(odf_ens + 0.5 * kodf1_ens, 0, None)
            odfi_ens /= odfi_ens.sum()
            # CALL 2/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse2 = np.dot(L, fsei) * dt
            kodf2 = dotodf * dt * epsnot
            kodf2_ens = dotodf_ens * dt * epsnot
            kac2 = dotacs * dt * epsnot
            kac2_ens = dotacs_ens * dt * epsnot
            fsei = fse + 0.5 * kfse2
            acsi = np.clip(acs + 0.5 * kac2, -1, 1)
            acsi_ens = np.clip(acs_ens + 0.5 * kac2_ens, -1, 1)
            odfi = np.clip(odf + 0.5 * kodf2, 0, None)
            odfi /= odfi.sum()
            odfi_ens = np.clip(odf_ens + 0.5 * kodf2_ens, 0, None)
            odfi_ens /= odfi_ens.sum()
            # CALL 3/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse3 = np.dot(L, fsei) * dt
            kodf3 = dotodf * dt * epsnot
            kodf3_ens = dotodf_ens * dt * epsnot
            kac3 = dotacs * dt * epsnot
            kac3_ens = dotacs_ens * dt * epsnot
            fsei = fse + kfse3
            acsi = np.clip(acs + kac3, -1, 1)
            acsi_ens = np.clip(acs_ens + kac3_ens, -1, 1)
            odfi = np.clip(odf + kodf3, 0, None)
            odfi /= odfi.sum()
            odfi_ens = np.clip(odf_ens + kodf3_ens, 0, None)
            odfi_ens /= odfi_ens.sum()
            # CALL 4/4
            dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
            )
            kfse4 = np.dot(L, fsei) * dt
            kodf4 = dotodf * dt * epsnot
            kodf4_ens = dotodf_ens * dt * epsnot
            kac4 = dotacs * dt * epsnot
            kac4_ens = dotacs_ens * dt * epsnot
            fse += (kfse1 / 2 + kfse2 + kfse3 + kfse4 / 2) / 3
            acs = np.clip(acs + (kac1 / 2 + kac2 + kac3 + kac4 / 2) / 3, -1, 1)
            acs_ens = np.clip(
                acs_ens + (kac1_ens / 2 + kac2_ens + kac3_ens + kac4_ens / 2) / 3, -1, 1
            )
            odf += (kodf1 / 2 + kodf2 + kodf3 + kodf4 / 2) / 3
            odf_ens += (kodf1_ens / 2 + kodf2_ens + kodf3_ens + kodf4_ens / 2) / 3
            odf /= odf.sum()
            odf_ens /= odf_ens.sum()
            # Grain-boundary sliding
            mask = odf < chi / size
            mask_ens = odf_ens < chi / size
            acs[mask, :, :] = acs0[mask, :, :]
            acs_ens[mask_ens, :, :] = acs0[mask_ens, :, :]
            odf[mask] = chi / size
            odf_ens[mask_ens] = chi / size
            odf /= odf.sum()
            odf_ens /= odf_ens.sum()
        currTime += nbIter * dt
    return fse, acs, acs_ens, odf, odf_ens


# strain using Scipy solve_ivp, requiring the new pathline
def strain(pathTime, pathDense, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        acs0 = dictGlobals['acs0']
        chi = dictGlobals['chi']
        gridCoords = dictGlobals['gridCoords']
        iDefMech = dictGlobals['iDefMech']
        size = dictGlobals['size']
    """

    def extractVars(y):
        fse = y[:9].copy().reshape(3, 3)
        acs = y[9 : size * 9 + 9].copy().reshape(size, 3, 3).clip(-1, 1)
        acs_ens = y[size * 9 + 9 : size * 18 + 9].copy().reshape(size, 3, 3)
        acs_ens.clip(-1, 1, out=acs_ens)
        odf = y[size * 18 + 9 : size * 19 + 9].copy().clip(0, None)
        odf /= odf.sum()
        odf_ens = y[size * 19 + 9 : size * 20 + 9].copy().clip(0, None)
        odf_ens /= odf_ens.sum()
        return fse, acs, acs_ens, odf, odf_ens

    def grainBoundarySliding(acs, acs_ens, odf, odf_ens):
        mask = odf < chi / size
        mask_ens = odf_ens < chi / size
        acs[mask, :, :] = acs0[mask, :, :]
        acs_ens[mask_ens, :, :] = acs0[mask_ens, :, :]
        odf[mask] = chi / size
        odf /= odf.sum()
        odf_ens[mask_ens] = chi / size
        odf_ens /= odf_ens.sum()
        return acs, acs_ens, odf, odf_ens

    def derivIVP(t, y):
        fsei, acsi, acsi_ens, odfi, odfi_ens = extractVars(y)
        dotacs, dotacs_ens, dotodf, dotodf_ens = deriv(
            L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha
        )
        return np.hstack(
            (
                np.dot(L, fsei).flatten(),
                dotacs.flatten() * epsnot,
                dotacs_ens.flatten() * epsnot,
                dotodf * epsnot,
                dotodf_ens * epsnot,
            )
        )

    def eventIVP(t, y):
        nonlocal alpha, e, epsnot, fse, L
        fse, acs, acs_ens, odf, odf_ens = extractVars(y)
        acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens, odf, odf_ens)
        y[9:] = np.hstack((acs.flatten(), acs_ens.flatten(), odf, odf_ens))
        currPoint = pathDense(t)
        L, epsnot = interpVelGrad(currPoint, dictGlobals)
        e = (L + L.transpose()) / 2
        alpha = iDefMech(*currPoint)
        return -1

    fse = np.identity(3)
    for time in reversed(pathTime):
        if isInside(pathDense(time), dictGlobals):
            currTime = time
            break
    currPoint = pathDense(currTime)
    L, epsnot = interpVelGrad(currPoint, dictGlobals)
    e = (L + L.transpose()) / 2
    alpha = iDefMech(*currPoint)

    currVel = interpVel(currPoint, dictGlobals)
    indLeft = [
        np.searchsorted(gridCoords[x], currPoint[x]) for x in range(currPoint.size)
    ]
    gridStep = []
    for coord, ind in zip(gridCoords, indLeft):
        try:
            gridStep.append(coord[ind + 1] - coord[ind])
        except IndexError:
            gridStep.append(coord[ind] - coord[ind - 1])
    dtPathline = min(gridStep) / 4 / norm(currVel, ord=2)
    dt = min(dtPathline, pathTime[0] - currTime, 1e-2 / epsnot)

    sol = solve_ivp(
        derivIVP,
        [currTime, pathTime[0]],
        np.hstack(
            (
                fse.flatten(),
                acs0.copy().flatten(),
                acs0.copy().flatten(),
                np.ones(size) / size,
                np.ones(size) / size,
            )
        ),
        method="RK45",
        first_step=dt,
        max_step=5e12,
        t_eval=[pathTime[0]],
        events=[eventIVP],
        atol=1e-8,
        rtol=1e-5,
    )
    fse, acs, acs_ens, odf, odf_ens = extractVars(sol.y.squeeze())
    acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens, odf, odf_ens)
    return fse, acs, acs_ens, odf, odf_ens


# Original eigen
# Find 3 eigenvalues of velocity gradient tensor
def eigen(L):
    Id = np.identity(3)
    Q = 0
    for ii in range(2):
        for jj in range(ii + 1, 3):
            Q -= (L[ii, ii] * L[jj, jj] - L[ii, jj] * L[jj, ii]) / 3
    R = np.linalg.det(L) / 2
    if abs(Q) < 1e-9:
        F = -np.ones((3, 3))
    elif Q**3 - R**2 >= 0:
        theta = np.arccos(R / Q**1.5)
        L1 = -2 * np.sqrt(Q) * np.cos(theta / 3)
        L2 = -2 * np.sqrt(Q) * np.cos((theta + 2 * np.pi) / 3)
        L3 = -2 * np.sqrt(Q) * np.cos((theta + 4 * np.pi) / 3)
        if abs(L1 - L2) < 1e-13:
            L1 = L2
        if abs(L3 - L2) < 1e-13:
            L3 = L2
        if abs(L1 - L3) < 1e-13:
            L1 = L3
        if L1 > L2 and (L1 > L3 or L2 == L3):
            F = np.dot(L - L2 * Id, L - L3 * Id)
        elif L2 > L3 and (L2 > L1 or L3 == L1):
            F = np.dot(L - L3 * Id, L - L1 * Id)
        elif L3 > L1 and (L3 > L2 or L1 == L2):
            F = np.dot(L - L1 * Id, L - L2 * Id)
        else:
            F = np.zeros((3, 3))
    else:
        xx = (np.sqrt(R**2 - Q**3) + abs(R)) ** (1 / 3)
        L1 = -(xx + Q / xx) if R == 0 else -np.sign(R) * (xx + Q / xx)
        L2 = L3 = -L1 / 2
        F = np.dot(L - L2 * Id, L - L3 * Id) if L1 > 1e-9 else np.zeros((3, 3))
    return F


# Original pipar
def pipar(currPoint, dictGlobals=None):
    # isacalc: Calculates ISA Orientation at grid point
    def isacalc(L, veloc):
        nonlocal GOL
        F = eigen(L)
        print(eigh(np.dot(F, np.transpose(F))), eigh(Ua))
        if np.sum(F) == -9:
            isa = -np.ones(3)
        elif np.sum(np.absolute(F)) == 0:
            isa = np.zeros(3)
            GOL = -1
        else:
            isa = eigh(np.dot(F, np.transpose(F)))[1][:, -1]
        return isa if dim == 3 else isa[::2]

    GOL = 10
    # local velocity vector and orientation of infinite strain axis
    veloc = interpVel(currPoint, dictGlobals)
    dt = 1e-2 / norm(veloc, ord=2)
    # previous point on the streamline
    prevPoint = currPoint - dt * veloc
    if isInside(prevPoint, dictGlobals) is False:
        return -10
    veloc = interpVel(prevPoint, dictGlobals)
    # epsnot: reference strain rate
    L, epsnot = interpVelGrad(prevPoint, dictGlobals)
    L /= epsnot
    # calculation of the ISA
    isa = veloc / norm(veloc, ord=2) if np.sum(isacalc(L)) == -3 else isacalc(L)
    # angle between ISA and flow direction
    thetaISA = np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa))
    # next point on the streamline
    nextPoint = currPoint + dt * veloc
    if isInside(nextPoint, dictGlobals) is False:
        return -10
    veloc = veloc / norm(veloc, ord=2)
    L, epsnot = interpVelGrad(nextPoint, dictGlobals)
    L /= epsnot
    # calculation of the ISA
    isa = veloc if np.sum(isacalc(L)) == -3 else isacalc(L)
    L, epsnot = interpVelGrad(currPoint, dictGlobals)
    # angle between ISA and flow direction
    thetaISA = abs(thetaISA - np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa)))
    if thetaISA > np.pi:
        thetaISA -= np.pi
    return min(GOL, thetaISA / 2 / dt / epsnot)


# New pipar, mainly a re-write of isacalc
def pipar(currPoint, dictGlobals=None):
    # Calculates ISA Orientation at grid point
    def isacalc(L, veloc):
        nonlocal GOL
        # Kaminski, Ribe & Browaeys (2004): Appendix B
        evals = eigvalsh(L)
        if np.sum(np.absolute(evals) < 1e-9) >= 2 - veloc.size % 2:
            assert (
                abs(evals[0] * evals[1] + evals[0] * evals[2] + evals[1] * evals[2])
                < 1e-9
            )
            return veloc / norm(veloc, ord=2)
        else:
            ind = np.argsort(np.absolute(evals))[-1]
            if np.isreal(evals[ind]):
                a = (ind + 1) % 3
                b = (ind + 2) % 3
                Id = np.identity(3)
                Fa = (
                    np.dot(L - evals[a] * Id, L - evals[b] * Id)
                    / (evals[ind] - evals[a])
                    / (evals[ind] - evals[b])
                )
                Ua = np.dot(np.transpose(Fa), Fa)
                return eigh(Ua)[1][:: 4 - veloc.size, -1]
            else:
                GOL = -1
                return np.zeros(veloc.size)

    GOL = 10
    # local velocity vector and orientation of infinite strain axis
    veloc = interpVel(currPoint, dictGlobals)
    dt = 1e-2 / norm(veloc, ord=2)
    # previous point on the streamline
    prevPoint = currPoint - dt * veloc
    if isInside(prevPoint, dictGlobals) is False:
        return -10
    veloc = interpVel(prevPoint, dictGlobals)
    # epsnot: reference strain rate
    L, epsnot = interpVelGrad(prevPoint, dictGlobals)
    L /= epsnot
    # calculation of the ISA
    isa = isacalc(L, veloc)
    # angle between ISA and flow direction
    thetaISA = np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa))
    # next point on the streamline
    nextPoint = currPoint + dt * veloc
    if isInside(nextPoint, dictGlobals) is False:
        return -10
    veloc = interpVel(nextPoint, dictGlobals)
    L, epsnot = interpVelGrad(nextPoint, dictGlobals)
    L /= epsnot
    # calculation of the ISA
    isa = isacalc(L, veloc)
    L, epsnot = interpVelGrad(currPoint, dictGlobals)
    # angle between ISA and flow direction
    thetaISA = abs(thetaISA - np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa)))
    if thetaISA > np.pi:
        thetaISA -= np.pi
    return min(GOL, thetaISA / 2 / dt / epsnot)


# Original pathline
def pathline(currPoint, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        gridCoords = dictGlobals['gridCoords']
        gridNodes = dictGlobals['gridNodes']
        iDefMech = dictGlobals['iDefMech']
    """
    stream_Lij = np.zeros((3, 3, 10000))
    stream_dt = np.zeros(10000)
    stream_e = np.zeros(10000)
    stream_alpha = np.zeros(10000)
    max_strain = 0
    step = -1
    # Construction of the streamline
    while max_strain < 10 and step < 1e4:
        step += 1
        currVel = interpVel(currPoint, dictGlobals)
        L, epsnot = interpVelGrad(currPoint, dictGlobals)
        dt = (
            min(
                abs(gridCoords[x][currInd[x] + 1] - gridCoords[x][currInd[x]])
                for x in range(len(currInd))
            )
            / 4
            / norm(currVel, ord=2)
        )
        # Record of the local velocity gradient tensor and time spent at that
        # point
        stream_Lij[:, :, step] = L
        stream_dt[step] = dt
        stream_e[step] = epsnot
        stream_alpha[step] = iDefMech(*currPoint)
        max_strain += dt * epsnot
        k1 = -currVel * dt
        newPoint = currPoint + 0.5 * k1
        if isInside(newPoint, dictGlobals) is False:
            break
        currVel = interpVel(newPoint, dictGlobals)
        k2 = -currVel * dt
        newPoint = currPoint + 0.5 * k2
        if isInside(newPoint, dictGlobals) is False:
            break
        currVel = interpVel(newPoint, dictGlobals)
        k3 = -currVel * dt
        newPoint = currPoint + k3
        if isInside(newPoint, dictGlobals) is False:
            break
        currVel = interpVel(newPoint, dictGlobals)
        currPoint += (k1 / 2 + k2 + k3 - currVel * dt / 2) / 3
        if isInside(currPoint, dictGlobals) is False:
            break
        for i in range(len(currPoint)):
            while gridCoords[i][currInd[i]] > currPoint[i] and currInd[i] > 0:
                currInd[i] -= 1
            while (
                gridCoords[i][currInd[i] + 1] < currPoint[i]
                and currInd[i] < gridNodes[i] - 2
            ):
                currInd[i] += 1
    return (
        step,
        stream_Lij[:, :, : step + 1],
        stream_e[: step + 1],
        stream_dt[: step + 1],
        stream_alpha[: step + 1],
    )


# New pathline, using Scipy solve_ivp
def pathline(currPoint, dictGlobals=None):
    def ivpFunc(time, pointCoords, dictGlobals=None):
        if isInside(pointCoords, dictGlobals):
            return interpVel(pointCoords, dictGlobals)
        else:
            return np.zeros(pointCoords.shape)

    def ivpJac(time, pointCoords, dictGlobals=None):
        if isInside(pointCoords, dictGlobals):
            L = interpVelGrad(pointCoords, dictGlobals, True)
            if pointCoords.size == 2:
                return np.array([[L[0, 0], L[0, 2]], [L[2, 0], L[2, 2]]])
            else:
                return L
        else:
            return np.zeros((pointCoords.size, pointCoords.size))

    def maxStrain(time, pointCoords, dictGlobals=None):
        nonlocal eventTime, eventTimePrev, eventStrain, eventStrainPrev, eventFlag
        if eventFlag:
            return (eventStrain if time == eventTime else eventStrainPrev) - 10
        elif isInside(pointCoords, dictGlobals):
            L, epsnot = interpVelGrad(pointCoords, dictGlobals)
            eventStrainPrev = eventStrain
            eventStrain += abs(time - eventTime) * epsnot
            if eventStrain >= 10:
                eventFlag = True
            eventTimePrev = eventTime
            eventTime = time
            return eventStrain - 10
        else:
            return 0

    maxStrain.terminal = True
    eventStrain = eventTime = 0
    eventStrainPrev = eventTimePrev = None
    eventFlag = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sol = solve_ivp(
            ivpFunc,
            [0, -100e6 * 365.25 * 8.64e4],
            currPoint,
            method="RK45",
            first_step=1e11,
            max_step=np.inf,
            t_eval=None,
            events=[maxStrain],
            args=(dictGlobals,),
            dense_output=True,
            jac=ivpJac,
            atol=1e-8,
            rtol=1e-5,
        )
    return sol.t, sol.sol


# DRex matching the original implementation of pathline and strain
def DRex(locInd, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        gridCoords = dictGlobals['gridCoords']
    """
    # Initial location of the grid point
    currCoords = np.array(
        [(coord[ind] + coord[ind + 1]) / 2 for coord, ind in zip(gridCoords, locInd)]
    )
    # Grain Orientation Lag
    GOL = pipar(currCoords, dictGlobals)
    # Backward calculation of the pathline for each tracer
    step, stream_Lij, stream_e, stream_dt, stream_alpha = pathline(
        list(locInd), currCoords, dictGlobals
    )
    # Inward calculation of the LPO
    # Random initial LPO
    Fij, odf, odf_ens, acs, acs_ens = strain(
        step, stream_Lij, stream_e, stream_dt, stream_alpha, dictGlobals
    )
    # Left-stretch tensor for FSE calculation
    LSij = np.dot(Fij, np.transpose(Fij))
    eigval, eigvects = eigh(LSij)
    # pick up the orientation of the long axis of the FSE
    phi_fse = np.arctan2(eigvects[-1, -1], eigvects[0, -1])
    # natural strain = ln(a / c) where a is the long axis = max(eigval) ** 0.5
    ln_fse = np.log(eigval[-1] / eigval[0]) / 2
    # Cijkl tensor (using Voigt average)
    Sav = voigt(odf, odf_ens, acs, acs_ens)
    # percentage anisotropy and orientation of hexagonal symmetry axis
    perc_a, phi_a, radani, percani = decsym(Sav)
    return locInd[::-1], radani, percani, GOL, ln_fse, phi_fse, phi_a, perc_a


# DRex matching the new implementation of pathline and strain
def DRex(locInd, dictGlobals=None):
    """Uncomment this block to use Ray
    if dictGlobals:
        gridCoords = dictGlobals['gridCoords']
    """
    # Initial location of the grid point
    currCoords = np.array(
        [(coord[ind] + coord[ind + 1]) / 2 for coord, ind in zip(gridCoords, locInd)]
    )
    # Grain Orientation Lag
    GOL = pipar(currCoords, dictGlobals)
    # Backward calculation of the pathline for each tracer
    pathTime, pathDense = pathline(currCoords, dictGlobals)
    # Inward calculation of the LPO | Random initial LPO
    Fij, odf, odf_ens, acs, acs_ens = strain(pathTime, pathDense, dictGlobals)
    # Left-stretch tensor for FSE calculation
    LSij = np.dot(Fij, np.transpose(Fij))
    eigval, eigvects = eigh(LSij)
    # pick up the orientation of the long axis of the FSE
    phi_fse = np.arctan2(eigvects[-1, -1], eigvects[0, -1])
    # natural strain = ln(a / c) where a is the long axis = max(eigval) ** 0.5
    ln_fse = np.log(eigval[-1] / eigval[0]) / 2
    # Cijkl tensor (using Voigt average)
    Sav = voigt(odf, odf_ens, acs, acs_ens)
    # percentage anisotropy and orientation of hexagonal symmetry axis
    perc_a, phi_a, radani, percani = decsym(Sav)
    return locInd[::-1], radani, percani, GOL, ln_fse, phi_fse, phi_a, perc_a
