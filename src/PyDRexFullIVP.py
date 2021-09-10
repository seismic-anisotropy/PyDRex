#!/usr/bin/env python3

###############################################################################
# Python 3 Version of D-Rex
###############################################################################

import argparse
from numba import jit
import numpy as np
from numpy.linalg import eigh, eigvalsh, norm
from scipy.integrate import solve_ivp, RK45
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R
from time import perf_counter
from vtk import vtkXMLUnstructuredGridReader, vtkXMLPUnstructuredGridReader
from warnings import catch_warnings, simplefilter

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
@jit(nopython=True)
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
    K = np.trace(dilatStifTens) / 9  # Bulk modulus
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
            adv = np.arccos(abs(sdv))
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
    # Rotate in SCCA
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
    # Form the 3x3 Christoffel tensor
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
    if len(pointCoords) == 2:
        L = np.array([[iLxx(*pointCoords), 0, -iLxz(*pointCoords)],
                      [0, 0, 0],
                      [-iLzx(*pointCoords), 0, iLzz(*pointCoords)]])
    else:
        iLxy, iLyx, iLyy, iLyz, iLzy = (
            dictGlobals['iLxy'], dictGlobals['iLyx'], dictGlobals['iLyy'],
            dictGlobals['iLyz'], dictGlobals['iLzy'])
        L = np.array(
            [[iLxx(*pointCoords), iLxy(*pointCoords), -iLxz(*pointCoords)],
             [iLyx(*pointCoords), iLyy(*pointCoords), -iLyz(*pointCoords)],
             [-iLzx(*pointCoords), -iLzy(*pointCoords), iLzz(*pointCoords)]])
    assert abs(np.trace(L)) < 1e-15
    if ivpIter:
        return L
    e = (L + np.transpose(L)) / 2
    epsnot = abs(eigvalsh(e)).max()
    return L, epsnot


# Rotation of a matrix line based on finite strain ellipsoid
@jit(nopython=True)
def fseDecomp(fse, ex, alt):
    # Left-strech tensor for finite-strain ellipsoid calculation
    leftStretchTens = np.dot(fse, np.transpose(fse))
    eigval, eigvect = eigh(leftStretchTens)
    rot = np.zeros(3)  # Induced spin
    for i, eigv in enumerate(eigval):
        otherEig = eigval[np.array([(i + 1) % 3, (i + 2) % 3])]
        if (np.abs((otherEig - eigv) / eigv) < 5e-2).any():
            return rot
    # Equation 34 Ribe (1992)
    H = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        H[i] = (eigval[i] * (eigval[j] - eigval[k]) / (eigval[i] - eigval[j])
                / (eigval[i] - eigval[k]))
    # Equation 33 Ribe (1992) | alt -> epsilon, ex -> E, eigvect -> a.T
    for i in range(3):
        for n in range(3):
            m = (n + 2) % 3
            for j in range(3):
                for k in range(3):
                    rot[i] += H[n] * ex[j, k] * (
                        eigvect[j, n] * eigvect[k, (n + 1) % 3]
                        * eigvect[i, m] + eigvect[j, m]
                        * np.dot(np.ascontiguousarray(alt[i, :, k]),
                                 np.ascontiguousarray(eigvect[:, m])))
    return rot


# Calculation of the rotation vector and slip rate
@jit(nopython=True, fastmath=True)
def deriv(lx, ex, acsi, acsi_ens, fse, odfi, odfi_ens, alpha):
    alt = np.zeros((3, 3, 3))  # Levi-Civita symbol
    for i in range(3):
        alt[i % 3, (i + 1) % 3, (i + 2) % 3] = 1
        alt[i % 3, (i + 2) % 3, (i + 1) % 3] = -1
    g, g_ens = np.empty((3, 3)), np.empty((3, 3))
    rt, rt_ens = np.zeros(size), np.empty(size)
    dotacs, dotacs_ens = np.zeros((size, 3, 3)), np.zeros((size, 3, 3))
    if alpha == 0:
        rotIni = fseDecomp(fse, ex, alt)
    else:
        rot, rot_ens = np.empty(3), np.empty(3)
    gam = np.empty(4)
    # Plastic deformation & Dynamic recrystallization
    # Update olivine fabric type in the loop?
    for i in range(size):
        # Calculate invariants for the four slip systems of olivine
        # Inline equation below Equation 5 Kaminski & Ribe (2001)
        bigi = np.zeros(4)
        for j in range(3):
            for k in range(3):
                bigi[0] += ex[j, k] * acsi[i, 0, j] * acsi[i, 1, k]
                bigi[1] += ex[j, k] * acsi[i, 0, j] * acsi[i, 2, k]
                bigi[2] += ex[j, k] * acsi[i, 2, j] * acsi[i, 1, k]
                bigi[3] += ex[j, k] * acsi[i, 2, j] * acsi[i, 0, k]
        # Equation 5 Kaminski & Ribe (2001)
        indInac, indMin, indInt, indMax = np.argsort(np.abs(bigi / tau))
        gam[indInac], gam[indMax] = 0, 1
        for index in [indMin, indInt]:
            frac = tau[indMax] / tau[index] * bigi[index] / bigi[indMax]
            gam[index] = frac * abs(frac) ** (stressexp - 1)
        # Equation 4 Kaminski & Ribe (2001)
        for j in range(3):
            for k in range(3):
                g[j, k] = 2 * (gam[0] * acsi[i, 0, j] * acsi[i, 1, k]
                               + gam[1] * acsi[i, 0, j] * acsi[i, 2, k]
                               + gam[2] * acsi[i, 2, j] * acsi[i, 1, k]
                               + gam[3] * acsi[i, 2, j] * acsi[i, 0, k])
                g_ens[j, k] = 2 * acsi_ens[i, 2, j] * acsi_ens[i, 0, k]
        # Equation 7 Kaminski & Ribe (2001)
        gamNum = gamNum_ens = gamDen = gamDen_ens = 0
        for j in range(3):
            k = (j + 1) % 3  # Fortran implementation uses j + 2 ???
            gamNum -= (lx[j, k] - lx[k, j]) * (g[j, k] - g[k, j])
            gamNum_ens -= (lx[j, k] - lx[k, j]) * (g_ens[j, k] - g_ens[k, j])
            # Mistake in the equation: kl+1 instead of kk+1 ???
            gamDen -= (g[j, k] - g[k, j]) ** 2
            gamDen_ens -= (g_ens[j, k] - g_ens[k, j]) ** 2
            for L in range(3):
                gamNum += 2 * g[j, L] * lx[j, L]
                gamNum_ens += 2 * g_ens[j, L] * lx[j, L]
                gamDen += 2 * g[j, L] ** 2
                gamDen_ens += 2 * g_ens[j, L] ** 2
        gam0 = gamNum / gamDen
        gam0_ens = gamNum_ens / gamDen_ens / tau_ens ** stressexp
        # Equation 22 Kaminski & Ribe (2001)
        for index in [indMin, indInt, indMax]:
            dislDens = (tau[index] ** (1.5 - stressexp)
                        * abs(gam[index] * gam0) ** (1.5 / stressexp))
            rt[i] += dislDens * np.exp(-lamb * dislDens ** 2)
        rt0_ens = (tau_ens ** (1.5 - stressexp)
                   * abs(gam0_ens) ** (1.5 / stressexp))
        rt_ens[i] = rt0_ens * np.exp(-lamb * rt0_ens ** 2)
        if alpha == 0:
            # Equation 33 Ribe (1992) and Equation 8 Kaminski & Ribe (2001) ???
            rot = rotIni.copy()
            for j in range(3):
                r, s = (j + 1) % 3, (j + 2) % 3
                rot[j] += (lx[s, r] - lx[r, s]) / 2
            rot_ens = rot.copy()
        else:
            # Equation 8 Kaminski & Ribe (2001)
            for j in range(3):
                r, s = (j + 1) % 3, (j + 2) % 3
                rot[j] = (lx[s, r] - lx[r, s] - (g[s, r] - g[r, s]) * gam0) / 2
                rot_ens[j] = (lx[s, r] - lx[r, s]
                              - (g_ens[s, r] - g_ens[r, s]) * gam0_ens) / 2
        # Equation 9 Kaminski & Ribe (2001)
        for p in range(3):
            for q in range(3):
                for r in range(3):
                    for s in range(3):
                        dotacs[i, p, q] += (
                            alt[q, r, s] * acsi[i, p, s] * rot[r])
                        dotacs_ens[i, p, q] += (
                            alt[q, r, s] * acsi_ens[i, p, s] * rot_ens[r])
    # Volume averaged energy
    Emean, Emean_ens = np.sum(odfi * rt), np.sum(odfi_ens * rt_ens)
    # Change of volume fraction by grain boundary migration
    dotodf = Xol * mob * odfi * (Emean - rt)
    dotodf_ens = (1 - Xol) * mob * odfi_ens * (Emean_ens - rt_ens)
    return dotacs, dotacs_ens, dotodf, dotodf_ens


# Calculation of strain along pathlines
def strain(pathTime, pathDense, dictGlobals):
    def extractVars(y):
        fse = y[:9].copy().reshape(3, 3)
        acs = y[9:size * 9 + 9].copy().reshape(size, 3, 3).clip(-1, 1)
        acs_ens = y[size * 9 + 9:size * 18 + 9].copy().reshape(size, 3, 3)
        acs_ens.clip(-1, 1, out=acs_ens)
        odf = y[size * 18 + 9:size * 19 + 9].copy().clip(0, None)
        odf /= odf.sum()
        odf_ens = y[size * 19 + 9:size * 20 + 9].copy().clip(0, None)
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
            L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi, odfi_ens, alpha)
        return np.hstack((np.dot(L, fsei).flatten(), dotacs.flatten() * epsnot,
                          dotacs_ens.flatten() * epsnot,
                          dotodf * epsnot, dotodf_ens * epsnot))

    def eventIVP(t, y):
        nonlocal alpha, e, epsnot, fse, L
        fse, acs, acs_ens, odf, odf_ens = extractVars(y)
        acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens, odf,
                                                          odf_ens)
        y[9:] = np.hstack((acs.flatten(), acs_ens.flatten(), odf, odf_ens))
        currPoint = pathDense(t)
        L, epsnot = interpVelGrad(currPoint, dictGlobals)
        e = (L + L.transpose()) / 2
        alpha = iDefMech(*currPoint)
        return -1

    chi, gridCoords, iDefMech, iCellAvgLen, size = (
        dictGlobals['chi'], dictGlobals['gridCoords'], dictGlobals['iDefMech'],
        dictGlobals['iCellAvgLen'], dictGlobals['size'])

    # Direction cosine matrix with uniformly distributed rotations.
    acs0 = R.random(size, random_state=1).as_matrix()

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
    indLeft = [np.searchsorted(gridCoords[x], currPoint[x])
               for x in range(currPoint.size)]
    gridStep = []
    for coord, ind in zip(gridCoords, indLeft):
        try:
            gridStep.append(coord[ind + 1] - coord[ind])
        except IndexError:
            gridStep.append(coord[ind] - coord[ind - 1])
    # dtPathline = max(min(gridStep), iCellAvgLen(*currPoint)) / 4 / norm(currVel)
    dtPathline = min(gridStep) / 4 / norm(currVel)
    dt = min(dtPathline, pathTime[0] - currTime, 1e-2 / epsnot)

    sol = RK45(
        derivIVP, currTime,
        np.hstack((fse.flatten(), acs0.copy().flatten(), acs0.copy().flatten(),
                   np.ones(size) / size, np.ones(size) / size)), pathTime[0],
        first_step=dt, max_step=1.25 * dt, atol=1e-6, rtol=1e-3)
    sol.step()
    fse, acs, acs_ens, odf, odf_ens = extractVars(sol.y)
    acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens, odf,
                                                      odf_ens)
    sol.y[9:] = np.hstack((acs.flatten(), acs_ens.flatten(), odf, odf_ens))
    while sol.status != 'finished':
        currPoint = pathDense(sol.t)
        L, epsnot = interpVelGrad(currPoint, dictGlobals)
        e = (L + L.transpose()) / 2
        alpha = iDefMech(*currPoint)

        currVel = interpVel(currPoint, dictGlobals)
        indLeft = [np.searchsorted(gridCoords[x], currPoint[x])
                   for x in range(currPoint.size)]
        gridStep = []
        for coord, ind in zip(gridCoords, indLeft):
            try:
                gridStep.append(coord[ind + 1] - coord[ind])
            except IndexError:
                gridStep.append(coord[ind] - coord[ind - 1])
        # dtPathline = max(min(gridStep), iCellAvgLen(*currPoint)) / 4 / norm(currVel)
        dtPathline = min(gridStep) / 4 / norm(currVel)
        dt = min(dtPathline, pathTime[0] - currTime, 1e-2 / epsnot)
        sol.max_step = 1.25 * dt

        sol.step()
        fse, acs, acs_ens, odf, odf_ens = extractVars(sol.y)
        acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens, odf,
                                                          odf_ens)
        sol.y[9:] = np.hstack((acs.flatten(), acs_ens.flatten(), odf, odf_ens))

    '''
    sol = solve_ivp(
        derivIVP, [currTime, pathTime[0]],
        np.hstack((fse.flatten(), acs0.copy().flatten(), acs0.copy().flatten(),
                   np.ones(size) / size, np.ones(size) / size)),
        method='DOP853', first_step=dt, max_step=dt * 5,
        t_eval=[pathTime[0]], events=[eventIVP], atol=1e-4, rtol=1e-3)
    '''
    fse, acs, acs_ens, odf, odf_ens = extractVars(sol.y.squeeze())
    acs, acs_ens, odf, odf_ens = grainBoundarySliding(acs, acs_ens,
                                                      odf, odf_ens)
    return fse, acs, acs_ens, odf, odf_ens


# Calculates elastic tensor Cav_{ijkl} for olivine
@jit(nopython=True)
def voigt(acs, acs_ens, odf, odf_ens):
    C0, C0_ens = np.empty((3, 3, 3, 3)), np.empty((3, 3, 3, 3))
    Cav, Sav = np.zeros((3, 3, 3, 3)), np.empty((6, 6))
    # Indices to form Cijkl from Sij
    ijkl = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype=np.int16)
    # Elastic tensors
    for p in range(3):
        for q in range(3):
            i = ijkl[p, q]
            for r in range(3):
                for s in range(3):
                    j = ijkl[r, s]
                    C0[p, q, r, s], C0_ens[p, q, r, s] = S0[i, j], S0_ens[i, j]
    for nu in range(size):
        C0rot = rotateTens(C0, acs[nu, ...].transpose())
        C0rot_ens = rotateTens(C0_ens, acs_ens[nu, ...].transpose())
        Cav += C0rot * odf[nu] * Xol + C0rot_ens * odf_ens[nu] * (1 - Xol)
    # Average stiffness matrix
    for i in range(6):
        p, q = (i + i // 3) % 3, (i + 2 * (i // 3)) % 3
        for j in range(6):
            r, s = (j + j // 3) % 3, (j + 2 * (j // 3)) % 3
            Sav[i, j] = Cav[p, q, r, s]
    return Sav


# Calculates GOL parameter at grid point
def pipar(currPoint, dictGlobals):
    # Calculates ISA Orientation at grid point
    def isacalc(L, veloc):
        nonlocal GOL
        # Kaminski, Ribe & Browaeys (2004): Appendix B
        evals = eigvalsh(L)
        if np.sum(abs(evals) < 1e-9) >= 2 - veloc.size % 2:
            assert abs(evals[0] * evals[1] + evals[0] * evals[2]
                       + evals[1] * evals[2]) < 1e-9
            return veloc / norm(veloc, ord=2)
        else:
            ind = np.argsort(abs(evals))[-1]
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
    with catch_warnings():
        simplefilter('ignore', category=UserWarning)
        sol = solve_ivp(ivpFunc, [0, -100e6 * 365.25 * 8.64e4], currPoint,
                        method='LSODA', first_step=1e10, max_step=np.inf,
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
        vtkReader = vtkXMLUnstructuredGridReader()
    elif inputExt == 'pvtu':
        vtkReader = vtkXMLPUnstructuredGridReader()
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

    nbCells = vtkOut.GetNumberOfCells()
    cellLen = np.zeros(coords.shape[0])
    cellAround = np.zeros(coords.shape[0])
    for i in range(nbCells):
        cell = vtkOut.GetCell(i)
        cellLength = np.sqrt(cell.GetLength2())
        for j in range(cell.GetNumberOfPoints()):
            pointId = cell.GetPointId(j)
            cellLen[pointId] += cellLength
            cellAround[pointId] += 1
    dictGlobals['iCellAvgLen'] = NearestNDInterpolator(
        coords[:, :-1], cellLen / cellAround)

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
            ray.init(address='auto', _redis_password=inputArgs.redis_pass)
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
