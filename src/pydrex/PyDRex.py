#!/usr/bin/env python3

###############################################################################
# Python 3 Version of D-Rex
###############################################################################

import argparse
from numba import jit
import numpy as np
from numpy.linalg import eigh, eigvalsh, norm
from scipy.integrate import solve_ivp
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R
from time import perf_counter
import vtk
import warnings

from DRexParam import (checkpoint, chi, gridCoords, gridMax, gridMin,
                       gridNodes, lamb, mob, name, size, stressexp, S0, S0_ens,
                       tau, tau_ens, Xol)

###############################################################################
#
# symFromUpper    : Generates a symmetric array from the upper triangle part of
#                   another array
# formElasticTens : Yields the 4th-order elastic tensor equivalent to the 6x6
#                   Voigt matrix
# formVoigtMat    : Yields the 6x6 Voigt matrix equivalent to the 4th-order
#                   elastic tensor
# formVoigtVec    : Yields the 21-component Voigt vector equivalent to the 6x6
#                   Voigt matrix
# rotateTens      : Rotates a 4th-order tensor
# trIsoProj       : Transverse isotropy projector
# scca            : Forms symmetric cartesian coordinate system
# calcAzimuth     : Calculates the azimuthal fast direction in a horizontal
#                   plane
# decsym          : Decomposition into transverse isotropy symmetry
# interpVel       : Interpolates the velocity vector at a given point
# interpVelGrad   : Interpolates the velocity gradient tensor at a given point
# fseDecomp       : Rotates a matrix line based on finite strain ellipsoid
# deriv           : Calculation of the rotation vector and slip rate
# strain          : Calculation of strain along pathlines
# voigt           : Calculates elastic tensor cav_{ijkl} for olivine
# pipar           : Calculates GOL parameter at grid point
# isacalc         : Calculates ISA Orientation at grid point
# pathline        : Determines a pathline given a position and a velocity field
# isInside        : Checks if a point lies within the numerical domain
# DRex            : Main function
# main            : Init function
#
###############################################################################


# Generates a symmetric array from the upper triangle part of another array
def symFromUpper(arr):
    return np.triu(arr) + np.transpose(np.triu(arr)) - np.diag(np.diag(arr))


# Yields the 4th-order elastic tensor equivalent to the 6x6 Voigt matrix
def formElasticTens(mat):
    # Equation 2.1 Browaeys and Chevrot (2004)
    tens = np.empty((3, 3, 3, 3))
    for p in range(3):
        for q in range(3):
            deltaPQ = 1 if p == q else 0
            i = (p + 1) * deltaPQ + (1 - deltaPQ) * (7 - p - q) - 1
            for r in range(3):
                for s in range(3):
                    deltaRS = 1 if r == s else 0
                    j = (r + 1) * deltaRS + (1 - deltaRS) * (7 - r - s) - 1
                    tens[p, q, r, s] = mat[i, j]
    return tens


# Yields the 6x6 Voigt matrix equivalent to the 4th-order elastic tensor
def formVoigtMat(tens):
    # Equation 2.1 Browaeys and Chevrot (2004)
    # Components c_{ijkl} which map to C_{IJ} and C_{JI} (Table 1) are averaged
    mat = np.zeros((6, 6))
    matInd = np.zeros((6, 6))
    for p in range(3):
        for q in range(3):
            deltaPQ = 1 if p == q else 0
            i = (p + 1) * deltaPQ + (1 - deltaPQ) * (7 - p - q) - 1
            for r in range(3):
                for s in range(3):
                    deltaRS = 1 if r == s else 0
                    j = (r + 1) * deltaRS + (1 - deltaRS) * (7 - r - s) - 1
                    mat[i, j] += tens[p, q, r, s]
                    matInd[i, j] += 1
    mat /= matInd
    return (mat + mat.transpose()) / 2


# Yields the 21-component Voigt vector equivalent to the 6x6 Voigt matrix
def formVoigtVec(mat):
    # Equation 2.2 Browaeys and Chevrot (2004)
    vec = np.zeros(21)
    for i in range(3):
        vec[i] = mat[i, i]
        vec[i + 3] = np.sqrt(2) * mat[(i + 1) % 3, (i + 2) % 3]
        vec[i + 6] = 2 * mat[i + 3, i + 3]
        vec[i + 9] = 2 * mat[i, i + 3]
        vec[i + 12] = 2 * mat[(i + 2) % 3, i + 3]
        vec[i + 15] = 2 * mat[(i + 1) % 3, i + 3]
        vec[i + 18] = 2 * np.sqrt(2) * mat[(i + 1) % 3 + 3, (i + 2) % 3 + 3]
    return vec


# Rotates a 4th-order tensor
def rotateTens(tens, rot):
    rotTens = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for L in range(3):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                for d in range(3):
                                    rotTens[i, j, k, L] += (
                                        rot[i, a] * rot[j, b] * rot[k, c]
                                        * rot[L, d] * tens[a, b, c, d])
    return rotTens


# Transverse isotropy projector
def trIsoProj(x):
    # Appendix A4 Browaeys and Chevrot (2004)
    # An elastic medium with hexagonal symmetry is equivalent to a transversely
    # isotropic medium
    y = np.zeros(21)
    y[0] = y[1] = 3 / 8 * (x[0] + x[1]) + x[5] / 4 / np.sqrt(2) + x[8] / 4
    y[2] = x[2]
    y[3] = y[4] = (x[3] + x[4]) / 2
    y[5] = ((x[0] + x[1]) / 4 / np.sqrt(2) + 3 / 4 * x[5]
            - x[8] / 2 / np.sqrt(2))
    y[6] = y[7] = (x[6] + x[7]) / 2
    y[8] = (x[0] + x[1]) / 4 - x[5] / 2 / np.sqrt(2) + x[8] / 2
    return norm(x - y, ord=2)


# Forms symmetric cartesian coordinate system
def scca(voigtMat):
    # Equation 3.4 Browaeys and Chevrot (2004)
    dilatStifTens = np.empty((3, 3))
    for i in range(3):
        dilatStifTens[i, i] = voigtMat[:3, i].sum()
    dilatStifTens[0, 1] = dilatStifTens[1, 0] = voigtMat[:3, 5].sum()
    dilatStifTens[0, 2] = dilatStifTens[2, 0] = voigtMat[:3, 4].sum()
    dilatStifTens[1, 2] = dilatStifTens[2, 1] = voigtMat[:3, 3].sum()
    # Equation 3.5 Browaeys and Chevrot (2004)
    voigtStifTens = np.empty((3, 3))
    voigtStifTens[0, 0] = voigtMat[0, 0] + voigtMat[4, 4] + voigtMat[5, 5]
    voigtStifTens[1, 1] = voigtMat[1, 1] + voigtMat[3, 3] + voigtMat[5, 5]
    voigtStifTens[2, 2] = voigtMat[2, 2] + voigtMat[3, 3] + voigtMat[4, 4]
    voigtStifTens[0, 1] = voigtMat[0, 5] + voigtMat[1, 5] + voigtMat[3, 4]
    voigtStifTens[0, 2] = voigtMat[0, 4] + voigtMat[2, 4] + voigtMat[3, 5]
    voigtStifTens[1, 2] = voigtMat[1, 3] + voigtMat[2, 3] + voigtMat[4, 5]
    voigtStifTens = symFromUpper(voigtStifTens)
    # Appendix A5 Browaeys and Chevrot (2004)
    K = np.trace(dilatStifTens) / 9  # Incompressibility modulus
    G = (np.trace(voigtStifTens) - 3 * K) / 10  # Shear modulus
    isoVec = np.hstack((np.repeat(K + 4 * G / 3, 3),
                        np.repeat(np.sqrt(2) * (K - 2 * G / 3), 3),
                        np.repeat(2 * G, 3), np.repeat(0, 12)))
    voigtVec = formVoigtVec(voigtMat)
    anisotropy = norm(voigtVec - isoVec)
    # Section 3.2 Browaeys and Chevrot (2004)
    eigVecDST = eigh(dilatStifTens)[1][:, ::-1]
    eigVecVST = eigh(voigtStifTens)[1][:, ::-1]
    # Search for SCCA directions
    for i in range(3):
        ndvc = 0
        advc = 10
        for j in range(3):
            sdv = np.clip(np.dot(eigVecDST[:, i], eigVecVST[:, j]), -1, 1)
            adv = np.arccos(np.absolute(sdv))
            if adv < advc:
                ndvc = int((j + 1) * np.sign(sdv)) if sdv != 0 else j + 1
                advc = adv
        eigVecDST[:, i] = (eigVecDST[:, i] + ndvc
                           * eigVecVST[:, abs(ndvc) - 1]) / 2
        eigVecDST[:, i] /= norm(eigVecDST[:, i], ord=2)
    # Higher symmetry axis
    elasTens = formElasticTens(voigtMat)
    normVec = norm(voigtVec)
    for i in range(3):
        dev = trIsoProj(formVoigtVec(formVoigtMat(rotateTens(
            elasTens,
            eigVecDST[:, [(i + j) % 3 for j in range(3)]].transpose()))))
        if dev < normVec:
            normVec = dev
            ndvc = i + 1
    # Rotate in sccs
    scc = eigVecDST[:, [(abs(ndvc) - 1 + i) % 3 for i in range(3)]].transpose()
    voigtVec = formVoigtVec(formVoigtMat(rotateTens(elasTens, scc)))
    return anisotropy, scc, voigtVec


# Calculates the azimuthal fast direction in a horizontal plane
def calcAzimuth(voigtMat, azi=0, inc=-np.pi / 2):
    # Code is adapted from the MATLAB Seismic Anisotropy Toolkit
    # https://github.com/andreww/MSAT/blob/master/msat/MS_phasevels.m
    # Create the cartesian vector
    Xr = np.array([np.cos(azi) * np.cos(inc),
                   -np.sin(azi) * np.cos(inc), np.sin(inc)])
    Xr /= norm(Xr)
    # Compute Eigenvector
    gamma = np.array([[Xr[0], 0, 0, 0, Xr[2], Xr[1]],
                      [0, Xr[1], 0, Xr[2], 0, Xr[0]],
                      [0, 0, Xr[2], Xr[1], Xr[0], 0]])
    T = np.dot(np.dot(gamma, voigtMat), gamma.transpose())
    S1 = eigh(T)[1][:, 1]
    # Calculate projection onto propagation plane
    S1P = np.cross(Xr, np.cross(Xr, S1))
    # Rotate into YZ plane to calculate angles
    rot1 = np.array([[np.cos(azi), np.sin(azi), 0],
                     [-np.sin(azi), np.cos(azi), 0], [0, 0, 1]])
    rot2 = np.array([[np.cos(inc), 0, -np.sin(inc)],
                     [0, 1, 0], [np.sin(inc), 0, np.cos(inc)]])
    VR = np.dot(np.dot(S1P, rot1), rot2)
    return -np.arctan2(VR[1], VR[2])


# Decomposition into transverse isotropy symmetry
def decsym(Sav):
    voigtMat = symFromUpper(Sav)
    # Original approach
    anisotropy, scc, voigtVec = scca(voigtMat)
    perc = (anisotropy - trIsoProj(voigtVec)) / norm(voigtVec) * 100
    tiAxis = scc[2, :]
    tiAxis /= norm(tiAxis)
    inclti = np.arcsin(tiAxis[2])
    # New approach
    radani = ((voigtMat[0, 0] + voigtMat[1, 1]) / 8 - voigtMat[0, 1] / 4
              + voigtMat[5, 5] / 2) / ((voigtMat[3, 3] + voigtMat[4, 4]) / 2)
    percani = np.sqrt(((voigtMat[4, 4] - voigtMat[3, 3]) / 2) ** 2
                      + voigtMat[4, 3] ** 2)
    azimuth = calcAzimuth(voigtMat)
    return perc, inclti, radani, percani, azimuth


# Interpolates the velocity vector at a given point
def interpVel(pointCoords, dictGlobals):
    iVelX, iVelZ = dictGlobals['iVelX'], dictGlobals['iVelZ']
    if len(pointCoords) == 3:
        iVelY = dictGlobals['iVelY']
    pointVel = np.array(
        [iVelX(*pointCoords),
         (iVelY(*pointCoords) if len(pointCoords) == 3 else np.nan),
         -iVelZ(*pointCoords)])
    return pointVel[~np.isnan(pointVel)]


# Interpolates the velocity gradient tensor at a given point
def interpVelGrad(pointCoords, dictGlobals, ivpIter=False):
    iLxx, iLxz, iLzx, iLzz = (dictGlobals['iLxx'], dictGlobals['iLxz'],
                              dictGlobals['iLzx'], dictGlobals['iLzz'])
    if len(pointCoords) == 3:
        iLxy, iLyx, iLyy, iLyz, iLzy = (
            dictGlobals['iLxy'], dictGlobals['iLyx'], dictGlobals['iLyy'],
            dictGlobals['iLyz'], dictGlobals['iLzy'])
    if len(pointCoords) == 2:
        L = np.array([[iLxx(*pointCoords), 0, -iLxz(*pointCoords)],
                      [0, 0, 0],
                      [-iLzx(*pointCoords), 0, iLzz(*pointCoords)]])
    else:
        L = np.array(
            [[iLxx(*pointCoords), iLxy(*pointCoords), -iLxz(*pointCoords)],
             [iLyx(*pointCoords), iLyy(*pointCoords), -iLyz(*pointCoords)],
             [-iLzx(*pointCoords), -iLzy(*pointCoords), iLzz(*pointCoords)]])
    assert np.abs(np.trace(L)) < 1e-15
    if ivpIter:
        return L
    e = (L + np.transpose(L)) / 2  # strain rate tensor
    epsnot = np.abs(eigvalsh(e)).max()  # reference strain rate
    return L, epsnot


# Rotation of a matrix line based on finite strain ellipsoid
@jit(nopython=True)
def fseDecomp(fse, ex, alt):
    # Left-strech tensor for fse calculation
    LSij = np.dot(fse, np.transpose(fse))
    eigval, eigvect = eigh(LSij)
    eigvect = np.transpose(eigvect)  # Each column of eigvect is an eigenvector
    # and its transpose is the cosines matrix
    rot = np.zeros(3)  # vector of induced spin omega_i, i=1,3
    for i, eigv in enumerate(eigval):
        otherEig = eigval[np.array([(i + 1) % 3, (i + 2) % 3])]
        if (np.absolute((otherEig - eigv) / eigv) < 5e-2).any():
            return rot
    H = np.zeros(3)  # ration H_n of the eigenvalues as in Ribe, 1992
    H[0] = (eigval[0] * (eigval[1] - eigval[2]) / (eigval[0] - eigval[1])
            / (eigval[0] - eigval[2]))
    H[1] = (eigval[1] * (eigval[2] - eigval[0]) / (eigval[1] - eigval[2])
            / (eigval[1] - eigval[0]))
    H[2] = (eigval[2] * (eigval[0] - eigval[1]) / (eigval[2] - eigval[0])
            / (eigval[2] - eigval[1]))
    # equation 33 of Ribe 1992 with nn=n, ii=i1, jj=i2, kk=i3, ll=i4
    for ii in range(3):
        for nn in range(3):
            int2 = 0
            for jj in range(3):
                for kk in range(3):
                    int1 = 0
                    for ll in range(3):
                        int1 += (alt[ii, ll, kk] * eigvect[(nn + 2) % 3, jj]
                                 * eigvect[(nn + 2) % 3, ll])
                    int2 += ((eigvect[nn, jj] * eigvect[(nn + 1) % 3, kk]
                              * eigvect[(nn + 2) % 3, ii] + int1) * ex[jj, kk])
            rot[ii] += H[nn] * int2
    return rot


# Calculation of the rotation vector and slip rate
@jit(nopython=True, fastmath=True)
def deriv(lx, ex, acsi, acsi_ens, fse, odfi, odfi_ens, alpha):
    alt = np.zeros((3, 3, 3))  # \epsilon_{ijk}
    for ii in range(3):
        alt[ii % 3, (ii + 1) % 3, (ii + 2) % 3] = 1
        alt[ii % 3, (ii + 2) % 3, (ii + 1) % 3] = -1
    g = np.zeros((3, 3))
    g_ens = np.zeros((3, 3))
    rt = np.zeros(size)
    dotacs = np.zeros((size, 3, 3))
    rt_ens = np.zeros(size)
    dotacs_ens = np.zeros((size, 3, 3))
    if alpha == 0:
        rotIni = fseDecomp(fse, ex, alt)
    else:
        rot = np.zeros(3)
        rot_ens = np.zeros(3)
    gam = np.zeros(4)
    # Plastic deformation & Dynamic recrystallization
    # Update olivine fabric type in the loop?
    for ii in range(size):
        # Calculate invariants for the four slip systems of olivine
        bigi = np.zeros(4)
        for jj in range(3):
            for kk in range(3):
                bigi[0] += ex[jj, kk] * acsi[ii, 0, jj] * acsi[ii, 1, kk]
                bigi[1] += ex[jj, kk] * acsi[ii, 0, jj] * acsi[ii, 2, kk]
                bigi[2] += ex[jj, kk] * acsi[ii, 2, jj] * acsi[ii, 1, kk]
                bigi[3] += ex[jj, kk] * acsi[ii, 2, jj] * acsi[ii, 0, kk]
        iinac, imin, iint, imax = np.argsort(np.absolute(bigi / tau))
        # Calculate weighting factors gam_s relative to value gam_i for
        # which I / tau is largest
        gam[iinac] = 0
        gam[imax] = 1
        rat = tau[imax] / bigi[imax]
        qint = rat * bigi[iint] / tau[iint]
        qmin = rat * bigi[imin] / tau[imin]
        sn1 = stressexp - 1
        gam[iint] = qint * abs(qint) ** sn1
        gam[imin] = qmin * abs(qmin) ** sn1
        # Calculation of G tensor
        for jj in range(3):
            for kk in range(3):
                g[jj, kk] = 2 * (gam[0] * acsi[ii, 0, jj] * acsi[ii, 1, kk]
                                 + gam[1] * acsi[ii, 0, jj] * acsi[ii, 2, kk]
                                 + gam[2] * acsi[ii, 2, jj] * acsi[ii, 1, kk]
                                 + gam[3] * acsi[ii, 2, jj] * acsi[ii, 0, kk])
                g_ens[jj, kk] = 2 * acsi_ens[ii, 2, jj] * acsi_ens[ii, 0, kk]
        # Calculation of strain rate on the softest slip system
        R1 = R2 = R1_ens = R2_ens = 0
        for jj in range(3):
            kk = (jj + 2) % 3
            R1 -= (g[jj, kk] - g[kk, jj]) ** 2
            R1_ens -= (g_ens[jj, kk] - g_ens[kk, jj]) ** 2
            R2 -= (g[jj, kk] - g[kk, jj]) * (lx[jj, kk] - lx[kk, jj])
            R2_ens -= ((g_ens[jj, kk] - g_ens[kk, jj])
                       * (lx[jj, kk] - lx[kk, jj]))
            for ll in range(3):
                R1 += 2 * g[jj, ll] ** 2
                R1_ens += 2 * g_ens[jj, ll] ** 2
                R2 += 2 * lx[jj, ll] * g[jj, ll]
                R2_ens += 2 * lx[jj, ll] * g_ens[jj, ll]
        gam0 = R2 / R1
        # Weight factor between olivine and enstatite
        gam0_ens = (R2_ens / R1_ens
                    * (1 / tau_ens) ** stressexp)
        # Dislocation density calculation
        rt1 = (tau[imax] ** (1.5 - stressexp)
               * abs(gam[imax] * gam0) ** (1.5 / stressexp))
        rt2 = (tau[iint] ** (1.5 - stressexp)
               * abs(gam[iint] * gam0) ** (1.5 / stressexp))
        rt3 = (tau[imin] ** (1.5 - stressexp)
               * abs(gam[imin] * gam0) ** (1.5 / stressexp))
        rt[ii] = (rt1 * np.exp(-lamb * rt1 ** 2)
                  + rt2 * np.exp(-lamb * rt2 ** 2)
                  + rt3 * np.exp(-lamb * rt3 ** 2))
        rt0_ens = (tau_ens ** (1.5 - stressexp)
                   * abs(gam0_ens) ** (1.5 / stressexp))
        rt_ens[ii] = rt0_ens * np.exp(-lamb * rt0_ens ** 2)
        # Calculation of the rotation rate:
        if alpha == 0:
            rot = rotIni.copy()
            rot[2] += (lx[1, 0] - lx[0, 1]) / 2
            rot[1] += (lx[0, 2] - lx[2, 0]) / 2
            rot[0] += (lx[2, 1] - lx[1, 2]) / 2
            rot_ens = rot.copy()
        else:
            rot[2] = (lx[1, 0] - lx[0, 1]) / 2 - (g[1, 0] - g[0, 1]) / 2 * gam0
            rot[1] = (lx[0, 2] - lx[2, 0]) / 2 - (g[0, 2] - g[2, 0]) / 2 * gam0
            rot[0] = (lx[2, 1] - lx[1, 2]) / 2 - (g[2, 1] - g[1, 2]) / 2 * gam0
            rot_ens[2] = ((lx[1, 0] - lx[0, 1]) / 2
                          - (g_ens[1, 0] - g_ens[0, 1]) / 2 * gam0_ens)
            rot_ens[1] = ((lx[0, 2] - lx[2, 0]) / 2
                          - (g_ens[0, 2] - g_ens[2, 0]) / 2 * gam0_ens)
            rot_ens[0] = ((lx[2, 1] - lx[1, 2]) / 2
                          - (g_ens[2, 1] - g_ens[1, 2]) / 2 * gam0_ens)
        # Derivative of the matrix of direction cosine
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        dotacs[ii, i1, i2] += (alt[i2, i3, i4]
                                               * acsi[ii, i1, i4] * rot[i3])
                        dotacs_ens[ii, i1, i2] += (alt[i2, i3, i4]
                                                   * acsi_ens[ii, i1, i4]
                                                   * rot_ens[i3])
    # Volume averaged energy
    Emean = np.sum(odfi * rt)
    Emean_ens = np.sum(odfi_ens * rt_ens)
    # Change of volume fraction by grain boundary migration
    dotodf = Xol * mob * odfi * (Emean - rt)
    dotodf_ens = (1 - Xol) * mob * odfi_ens * (Emean_ens - rt_ens)
    return dotacs, dotacs_ens, dotodf, dotodf_ens


# Calculation of strain along pathlines
def strain(pathTime, pathDense, dictGlobals):
    chi, gridCoords, iDefMech, size = (
        dictGlobals['chi'], dictGlobals['gridCoords'], dictGlobals['iDefMech'],
        dictGlobals['size'])
    fse = np.identity(3)
    # Direction cosine matrix with uniformly distributed rotations.
    acs0 = R.random(size, random_state=1).as_matrix()
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
        indLeft = [np.searchsorted(gridCoords[x], currPoint[x])
                   for x in range(currPoint.size)]
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
                L / epsnot, e / epsnot, acs, acs_ens, fse, odf, odf_ens,
                alpha)
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
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens,
                alpha)
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
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens,
                alpha)
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
                L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens,
                alpha)
            kfse4 = np.dot(L, fsei) * dt
            kodf4 = dotodf * dt * epsnot
            kodf4_ens = dotodf_ens * dt * epsnot
            kac4 = dotacs * dt * epsnot
            kac4_ens = dotacs_ens * dt * epsnot
            fse += (kfse1 / 2 + kfse2 + kfse3 + kfse4 / 2) / 3
            acs = np.clip(acs + (kac1 / 2 + kac2 + kac3 + kac4 / 2) / 3, -1, 1)
            acs_ens = np.clip(acs_ens + (kac1_ens / 2 + kac2_ens + kac3_ens
                                         + kac4_ens / 2) / 3, -1, 1)
            odf += (kodf1 / 2 + kodf2 + kodf3 + kodf4 / 2) / 3
            odf_ens += (kodf1_ens / 2 + kodf2_ens +
                        kodf3_ens + kodf4_ens / 2) / 3
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


# Calculates elastic tensor cav_{ijkl} for olivine
@jit(nopython=True)
def voigt(acs, acs_ens, odf, odf_ens):
    C0, C03 = np.zeros((3, 3, 3, 3)), np.zeros((3, 3, 3, 3))
    Cav, Sav = np.zeros((3, 3, 3, 3)), np.zeros((6, 6))
    # Indices to form Cijkl from Sij
    ijkl = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype=np.int16)
    # Elastic tensors c0_{ijkl}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ll in range(3):
                    C0[i, j, k, ll] = S0[ijkl[i, j], ijkl[k, ll]]
                    C03[i, j, k, ll] = S0_ens[ijkl[i, j], ijkl[k, ll]]
    for nu in range(size):
        Cav2, Cav3 = np.zeros((3, 3, 3, 3)), np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for ll in range(3):
                        for p in range(3):
                            for q in range(3):
                                for r in range(3):
                                    for ss in range(3):
                                        Cav2[i, j, k, ll] += (
                                            acs[nu, p, i] * acs[nu, q, j]
                                            * acs[nu, r, k] * acs[nu, ss, ll]
                                            * C0[p, q, r, ss])
                                        Cav3[i, j, k, ll] += (
                                            acs_ens[nu, p, i]
                                            * acs_ens[nu, q, j]
                                            * acs_ens[nu, r, k]
                                            * acs_ens[nu, ss, ll]
                                            * C03[p, q, r, ss])
                        Cav[i, j, k, ll] += Cav2[i, j, k, ll] * odf[nu] * Xol
                        Cav[i, j, k, ll] += (Cav3[i, j, k, ll] * odf_ens[nu]
                                             * (1 - Xol))
    # Indices to form Sij from Cijkl
    l1 = np.array([0, 1, 2, 1, 2, 0], dtype=np.int16)
    l2 = np.array([0, 1, 2, 2, 0, 1], dtype=np.int16)
    # Average stiffness matrix
    for i in range(6):
        for j in range(6):
            Sav[i, j] = Cav[l1[i], l2[i], l1[j], l2[j]]
    return Sav


# Calculates GOL parameter at grid point
def pipar(currPoint, dictGlobals):
    # Calculates ISA Orientation at grid point
    def isacalc(L, veloc):
        nonlocal GOL
        # Kaminski, Ribe & Browaeys (2004): Appendix B
        evals = eigvalsh(L)
        if np.sum(np.absolute(evals) < 1e-9) >= 2 - veloc.size % 2:
            assert abs(evals[0] * evals[1] + evals[0] * evals[2]
                       + evals[1] * evals[2]) < 1e-9
            return veloc / norm(veloc, ord=2)
        else:
            ind = np.argsort(np.absolute(evals))[-1]
            if np.isreal(evals[ind]):
                a, b = (ind + 1) % 3, (ind + 2) % 3
                Id = np.identity(3)
                Fa = (np.dot(L - evals[a] * Id, L - evals[b] * Id)
                      / (evals[ind] - evals[a]) / (evals[ind] - evals[b]))
                Ua = np.dot(np.transpose(Fa), Fa)
                return eigh(Ua)[1][::4 - veloc.size, -1]
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
    thetaISA = abs(thetaISA
                   - np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa)))
    if thetaISA > np.pi:
        thetaISA -= np.pi
    return min(GOL, thetaISA / 2 / dt / epsnot)


# Determines a pathline given a position and a velocity field
def pathline(currPoint, dictGlobals):
    def ivpFunc(time, pointCoords, dictGlobals):
        if isInside(pointCoords, dictGlobals):
            return interpVel(pointCoords, dictGlobals)
        else:
            return np.zeros(pointCoords.shape)

    def ivpJac(time, pointCoords, dictGlobals):
        if isInside(pointCoords, dictGlobals):
            L = interpVelGrad(pointCoords, dictGlobals, True)
            if pointCoords.size == 2:
                return np.array([[L[0, 0], L[0, 2]], [L[2, 0], L[2, 2]]])
            else:
                return L
        else:
            return np.zeros((pointCoords.size, pointCoords.size))

    def maxStrain(time, pointCoords, dictGlobals):
        nonlocal eventTime, eventTimePrev, eventStrain, eventStrainPrev, \
            eventFlag
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
        warnings.simplefilter('ignore', category=UserWarning)
        sol = solve_ivp(ivpFunc, [0, -100e6 * 365.25 * 8.64e4], currPoint,
                        method='RK45', first_step=1e10, max_step=np.inf,
                        t_eval=None, events=[maxStrain], args=(dictGlobals,),
                        dense_output=True, jac=ivpJac, atol=1e-8, rtol=1e-5)
    return sol.t, sol.sol


# Checks if a point lies within the numerical domain
def isInside(point, dictGlobals):
    gridMax, gridMin = dictGlobals['gridMax'], dictGlobals['gridMin']
    inside = True
    for coord, minBound, maxBound in zip(point, gridMin, gridMax):
        if coord < minBound or coord > maxBound:
            inside = False
            break
    return inside


def DRex(locInd, dictGlobals):
    gridCoords = dictGlobals['gridCoords']
    # Initial location of the grid point
    currCoords = np.array([(coord[ind] + coord[ind + 1]) / 2
                           for coord, ind in zip(gridCoords, locInd)])
    # Grain Orientation Lag
    GOL = pipar(currCoords, dictGlobals)
    # Backward calculation of the pathline for each tracer
    pathTime, pathDense = pathline(currCoords, dictGlobals)
    # Inward calculation of the LPO | Random initial LPO
    Fij, acs, acs_ens, odf, odf_ens = strain(pathTime, pathDense, dictGlobals)
    # Left-stretch tensor for FSE calculation
    LSij = np.dot(Fij, np.transpose(Fij))
    eigval, eigvect = eigh(LSij)
    # pick up the orientation of the long axis of the FSE
    if len(locInd) == 2:
        phi_fse = np.arctan2(eigvect[-1, -1], eigvect[0, -1])
    else:
        phi_fse = (np.arctan2(eigvect[-1, -1], norm(eigvect[:-1, -1]))
                   * np.sign(eigvect[:-1, -1].prod()))
    # natural strain = ln(a / c) where a is the long axis = max(eigval) ** 0.5
    ln_fse = np.log(eigval[-1] / eigval[0]) / 2
    # Cijkl tensor (using Voigt average)
    Sav = voigt(acs, acs_ens, odf, odf_ens)
    # percentage anisotropy and orientation of hexagonal symmetry axis
    perc_a, phi_a, radani, percani, azimuth = decsym(Sav)
    return (locInd[::-1], radani, percani, GOL, ln_fse, phi_fse, phi_a, perc_a,
            azimuth)


def main(inputArgs):
    if isinstance(inputArgs, list):
        inputArgs = args

    inputExt = inputArgs.input.split('.')[-1]
    if inputExt == 'vtu':
        vtkReader = vtk.vtkXMLUnstructuredGridReader()
    elif inputExt == 'pvtu':
        vtkReader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        raise RuntimeError(
            '''Please either provide a supported file format or implement the
    reader your require.''') from None

    vtkReader.SetFileName(inputArgs.input)
    vtkReader.Update()
    vtkOut = vtkReader.GetOutput()
    vtkData = vtkOut.GetPoints().GetData()
    coords = np.array([vtkData.GetTuple3(tup)
                       for tup in range(vtkData.GetNumberOfTuples())])
    if np.amin(coords[:, 1]) == np.amax(coords[:, 1]):
        raise RuntimeError(
            '''Please either provide a 2-D or 3-D dataset.''') from None
    elif np.amin(coords[:, 2]) == np.amax(coords[:, 2]):
        dim = 2
        coords[:, 1] = np.amax(coords[:, 1]) - coords[:, 1]
    else:
        dim = 3
        coords[:, 2] = np.amax(coords[:, 2]) - coords[:, 2]
    vtkScalars = vtkOut.GetPointData().GetScalars('DeformationMechanism')
    defMech = np.array([vtkScalars.GetTuple1(tup)
                        for tup in range(vtkScalars.GetNumberOfTuples())])
    vtkScalars = vtkOut.GetPointData().GetScalars('Velocity')
    vel = np.array([vtkScalars.GetTuple3(tup)
                    for tup in range(vtkScalars.GetNumberOfTuples())])
    vtkArray = vtkOut.GetPointData().GetArray('VelocityGradient')
    velGrad = np.array([vtkArray.GetTuple9(tup)
                        for tup in range(vtkArray.GetNumberOfTuples())])
    del vtkReader, vtkData, vtkScalars, vtkArray

    dictGlobals = {
        'chi': chi, 'gridMin': gridMin, 'gridMax': gridMax, 'size': size,
        'gridNodes': gridNodes, 'gridCoords': gridCoords}

    if dim == 2:
        interpObj = [['iVelX', 'iVelZ'], ['iLxx', 'iLzx', 'iLxz', 'iLzz']]
        if inputArgs.mpl:
            triIDs = []
            for i in range(vtkOut.GetNumberOfCells()):
                assert vtkOut.GetCell(i).GetNumberOfPoints() == 3
                IDsList = vtkOut.GetCell(i).GetPointIds()
                triIDs.append([IDsList.GetId(j)
                               for j in range(IDsList.GetNumberOfIds())])
            tri = Triangulation(coords[:, 0], coords[:, 1], triangles=triIDs)
            for objList, fieldVar in zip(interpObj, [vel, velGrad]):
                for i, objName in enumerate(objList):
                    dictGlobals[objName] = CubicTriInterpolator(
                        tri, fieldVar[:, i % 2 + i // 2 * 3],
                        kind=inputArgs.mpl)
        else:
            tri = Delaunay(coords[:, :-1])
            for objList, fieldVar in zip(interpObj, [vel, velGrad]):
                for i, objName in enumerate(objList):
                    dictGlobals[objName] = CloughTocher2DInterpolator(
                        tri, fieldVar[:, i % 2 + i // 2 * 3])
        dictGlobals['iDefMech'] = NearestNDInterpolator(coords[:, :-1],
                                                        defMech)
        del tri, vtkOut
    else:
        interpObj = [['iVelX', 'iVelY', 'iVelZ'],
                     ['iLxx', 'iLyx', 'iLzx', 'iLxy', 'iLyy', 'iLzy',
                      'iLxz', 'iLyz', 'iLzz']]
        for objList, fieldVar in zip(interpObj, [vel, velGrad]):
            for i, objName in enumerate(objList):
                dictGlobals[objName] = NearestNDInterpolator(
                    coords, fieldVar[:, i])
        dictGlobals['iDefMech'] = NearestNDInterpolator(coords, defMech)

    outVar = ['radani', 'percani', 'GOL', 'ln_fse', 'phi_fse', 'phi_a',
              'perc_a', 'azimuth']
    varDict = {}
    if inputArgs.restart:
        varDict.update(np.load(inputArgs.restart))
    else:
        arrDim = [x - 1 for x in reversed(gridNodes)]
        for var in outVar:
            varDict[var] = np.zeros(arrDim)
        varDict['indArr'] = np.zeros(arrDim[::-1])
        varDict['nodesComplete'] = 0

    if inputArgs.charm:  # Charm4Py
        nBatch = np.ceil(np.sum(varDict['indArr'] == 0) / 6e4).astype(int)
        for batch in range(nBatch):
            nodes2do = np.asarray(varDict['indArr'] == 0).nonzero()
            futures = charm.pool.map_async(
                partial(DRex, dictGlobals=dictGlobals),
                list(zip(*[nodes[:60_000] for nodes in nodes2do])),
                multi_future=True)
            for future in charm.iwait(futures):
                output = future.get()
                for i, var in enumerate(outVar):
                    varDict[var][output[0]] = output[i + 1]
                varDict['indArr'][output[0][::-1]] = 1
                varDict['nodesComplete'] += 1
                if varDict['nodesComplete'] % checkpoint == 0:
                    np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                             f"{varDict['nodesComplete']}", **varDict)
    elif inputArgs.ray:  # Ray
        if inputArgs.redis_pass:
            # Cluster execution
            ray.init(address='auto')
        else:
            # Single machine | Set local_mode to True to force single process
            ray.init(num_cpus=inputArgs.cpus, local_mode=False)

        dictGlobalsID = ray.put(dictGlobals)

        nBatch = np.ceil(np.sum(varDict['indArr'] == 0) / 6e4).astype(int)
        for batch in range(nBatch):
            nodes2do = np.asarray(varDict['indArr'] == 0).nonzero()
            futures = [DRex.remote(i, dictGlobalsID)
                       for i in zip(*[nodes[:60_000] for nodes in nodes2do])]
            while len(futures):
                readyIds, futures = ray.wait(
                    futures, num_returns=min([checkpoint, len(futures)]))
                for output in ray.get(readyIds):
                    for i, var in enumerate(outVar):
                        varDict[var][output[0]] = output[i + 1]
                    varDict['indArr'][output[0][::-1]] = 1
                    varDict['nodesComplete'] += 1
                del readyIds
                np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                         f"{varDict['nodesComplete']}", **varDict)

        ray.shutdown()
    else:  # Multiprocessing
        if __name__ == '__main__':
            with Pool(processes=inputArgs.cpus) as pool:
                for output in pool.imap_unordered(
                        partial(DRex, dictGlobals=dictGlobals),
                        zip(*np.asarray(varDict['indArr'] == 0).nonzero())):
                    for i, var in enumerate(outVar):
                        varDict[var][output[0]] = output[i + 1]
                    varDict['indArr'][output[0][::-1]] = 1
                    varDict['nodesComplete'] += 1
                    if varDict['nodesComplete'] % checkpoint == 0:
                        np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                                 f"{varDict['nodesComplete']}", **varDict)

    np.savez(f'PyDRex{dim}D_{name}_Complete', **varDict)

    end = perf_counter()
    hours = int((end - begin) / 3600)
    minutes = int((end - begin - hours * 3600) / 60)
    seconds = end - begin - hours * 3600 - minutes * 60
    print(f'{hours}:{minutes}:{seconds}')

    if inputArgs.charm:
        exit()


begin = perf_counter()

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''
Python 3 implementation of the seismic anisotropy software D-Rex. Parallel
execution is achieved through multiprocessing; for multi-node parallelism
enable either Ray or Charm4Py. The module DRexParam.py must be in the same
directory.''')
parser.add_argument('input', help='input file (expects a vtkUnstructuredGrid)')
parser.add_argument('--charm', action='store_true', help='''enables parallel
                    execution through Charm4Py''')
parser.add_argument('--cpus', help='''Number of CPUs requested; required when
                    executing on a single-node environment (except for
                    Charm4Py)''', metavar='', type=int)
parser.add_argument('--mpl', help='''Must be either 'geom' or 'min_E'; only
                    applies to 2-D simulations for which a mesh triangulation
                    already exists. If provided, velocity and velocity gradient
                    fields are interpolated using Matplotlib
                    CubicTriInterpolator instead of Scipy
                    CloughTocher2DInterpolator.''', metavar='')
parser.add_argument('--ray', action='store_true', help='''enables parallel
                    execution through Ray''')
parser.add_argument('--redis-pass', help='''Redis password; required when using
                    Ray on a multi-node environment''', metavar='')
parser.add_argument('--restart', help='''checkpoint file to restart from
                    (expects a Numpy NpzFile)''', metavar='')
args = parser.parse_args()

if args.charm and args.cpus:
    print('''Warning: --cpus is redundant when using Charm4Py; consider
          removing it.''')

if (args.ray and args.redis_pass) and args.cpus:
    print('''Warning: --cpus is redundant when using Ray in a multi-node
          envrionment; consider removing it.''')

if args.charm or args.ray:
    assert (args.charm and args.ray) is False

if args.charm:
    from charm4py import charm
elif args.ray:
    import ray
    DRex = ray.remote(DRex)
else:
    from multiprocessing import Pool

if args.ray is False:
    from functools import partial

if args.mpl:
    from matplotlib.tri import CubicTriInterpolator, Triangulation
else:
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.spatial import Delaunay

if args.charm:
    charm.start(main)
else:
    main(args)
