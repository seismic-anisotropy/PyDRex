#!/usr/bin/env python3

###############################################################################
# Python 3 Version of D-Rex
###############################################################################

# Libraries and functions imported
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
# symFromUpper  : Generates a symmetric array from the upper triangle part of
#                 another array
# mat2tens      : Yields the 4th-order tensor equivalent of a 6x6 matrix
# tens2mat      : Yields the 6x6 matrix equivalent of a 4th-order tensor
# mat2vec       : Yields the 21-component vector equivalent of a 6x6 matrix
# rotateTens    : Rotates a 4th-order tensor
# trIsoProj     : Transverse isotropy projector
# scca          : Forms symmetric cartesian system
# cal_azimuth   : Calculates the azimuthal fast direction in a horizontal plane
#                 (in degrees)
# decsym        : Decomposition into transverse isotropy tensor
# interpVel     : Interpolates the velocity vector at a given point
# interpVelGrad : Interpolates the velocity gradient tensor at a given point
# fseDecomp     : Rotates a matrix line based on finite strain ellipsoid
# deriv         : Calculation of the rotation vector and slip rate
# strain        : Calculation of strain along pathlines
# voigt         : Calculates elastic tensor cav_{ijkl} for olivine
# pipar         : Calculates GOL parameter at grid point
# isacalc       : Calculates ISA Orientation at grid point
# pathline      : Determines a pathline given a position and a velocity field
# isInside      : Checks if a point lies within the numerical domain
# DRex          : Main function
# main          : Init function
#
###############################################################################


# Generates a symmetric array from the upper triangle part of another array
def symFromUpper(arr):
    return np.triu(arr) + np.transpose(np.triu(arr)) - np.diag(np.diag(arr))


# Yields the 4th-order tensor equivalent of a 6x6 matrix
def mat2tens(mat):
    tens = np.zeros((3, 3, 3, 3))
    for ii in range(3):
        for jj in range(3):
            deltaIJ = 1 if ii == jj else 0
            p = deltaIJ * (ii + 1) + (1 - deltaIJ) * (7 - ii - jj) - 1
            for kk in range(3):
                for ll in range(3):
                    deltaKL = 1 if kk == ll else 0
                    q = deltaKL * (kk + 1) + (1 - deltaKL) * (7 - kk - ll) - 1
                    tens[ii, jj, kk, ll] = mat[p, q]
    return tens


# Yields the 6x6 matrix equivalent of a 4th-order tensor
def tens2mat(tens):
    mat = np.zeros((6, 6))
    for ii in range(3):
        mat[ii, ii] = tens[ii, ii, ii, ii]
        mat[ii, 3] = (tens[ii, ii, 1, 2] + tens[2, 1, ii, ii] +
                      tens[ii, ii, 2, 1] + tens[1, 2, ii, ii]) / 4
        mat[ii, 4] = (tens[ii, ii, 0, 2] + tens[2, 0, ii, ii] +
                      tens[ii, ii, 2, 0] + tens[0, 2, ii, ii]) / 4
        mat[ii, 5] = (tens[ii, ii, 0, 1] + tens[1, 0, ii, ii] +
                      tens[ii, ii, 1, 0] + tens[0, 1, ii, ii]) / 4
    for ii in range(2):
        mat[0, ii + 1] = (tens[0, 0, ii + 1, ii + 1] +
                          tens[ii + 1, ii + 1, 0, 0]) / 2
    mat[1, 2] = (tens[1, 1, 2, 2] + tens[2, 2, 1, 1]) / 2
    mat[3, 3] = (tens[1, 2, 1, 2] + tens[1, 2, 2, 1] +
                 tens[2, 1, 1, 2] + tens[2, 1, 2, 1]) / 4
    mat[4, 4] = (tens[0, 2, 0, 2] + tens[0, 2, 2, 0] +
                 tens[2, 0, 0, 2] + tens[2, 0, 2, 0]) / 4
    mat[5, 5] = (tens[1, 0, 1, 0] + tens[1, 0, 0, 1] +
                 tens[0, 1, 1, 0] + tens[0, 1, 0, 1]) / 4
    mat[3, 4] = (tens[1, 2, 0, 2] + tens[1, 2, 2, 0] +
                 tens[2, 1, 0, 2] + tens[2, 1, 2, 0] +
                 tens[0, 2, 1, 2] + tens[0, 2, 2, 1] +
                 tens[2, 0, 1, 2] + tens[2, 0, 2, 1]) / 8
    mat[3, 5] = (tens[1, 2, 0, 1] + tens[1, 2, 1, 0] +
                 tens[2, 1, 0, 1] + tens[2, 1, 1, 0] +
                 tens[0, 1, 1, 2] + tens[0, 1, 2, 1] +
                 tens[1, 0, 1, 2] + tens[1, 0, 2, 1]) / 8
    mat[4, 5] = (tens[0, 2, 0, 1] + tens[0, 2, 1, 0] +
                 tens[2, 0, 0, 1] + tens[2, 0, 1, 0] +
                 tens[0, 1, 0, 2] + tens[0, 1, 2, 0] +
                 tens[1, 0, 0, 2] + tens[1, 0, 2, 0]) / 8
    return symFromUpper(mat)


# Yields the 21-component vector equivalent of a 6x6 matrix
def mat2vec(mat):
    vec = np.zeros(21)
    for ii in range(3):
        vec[ii] = mat[ii, ii]
        vec[ii + 6] = 2 * mat[ii + 3, ii + 3]
        vec[ii + 9] = 2 * mat[ii, ii + 3]
        vec[ii + 12] = 2 * mat[(ii + 2) % 3, ii + 3]
        vec[ii + 15] = 2 * mat[(ii + 1) % 3, ii + 3]
    vec[3] = np.sqrt(2) * mat[1, 2]
    vec[4] = np.sqrt(2) * mat[0, 2]
    vec[5] = np.sqrt(2) * mat[0, 1]
    vec[18] = 2 * np.sqrt(2) * mat[4, 5]
    vec[19] = 2 * np.sqrt(2) * mat[3, 5]
    vec[20] = 2 * np.sqrt(2) * mat[3, 4]
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
    y = np.zeros(21)
    y[0] = y[1] = 3 / 8 * (x[0] + x[1]) + x[5] / 4 / np.sqrt(2) + x[8] / 4
    y[2] = x[2]
    y[3] = y[4] = (x[3] + x[4]) / 2
    y[5] = ((x[0] + x[1]) / 4 / np.sqrt(2) + 3 / 4 * x[5]
            - x[8] / 2 / np.sqrt(2))
    y[6] = y[7] = (x[6] + x[7]) / 2
    y[8] = (x[0] + x[1]) / 4 - x[5] / 2 / np.sqrt(2) + x[8] / 2
    return norm(x - y, ord=2)


# Forms symmetric Cartesian system
def scca(CE1, EL1, XN, XEC):
    DI = np.zeros((3, 3))
    VO = np.zeros((3, 3))
    for i in range(3):
        DI[i, i] = CE1[i, :3].sum()
    DI[0, 1] = CE1[5, :3].sum()
    DI[0, 2] = CE1[4, :3].sum()
    DI[1, 2] = CE1[3, :3].sum()
    DI = symFromUpper(DI)
    VO[0, 0] = CE1[0, 0] + CE1[5, 5] + CE1[4, 4]
    VO[1, 1] = CE1[5, 5] + CE1[1, 1] + CE1[3, 3]
    VO[2, 2] = CE1[4, 4] + CE1[3, 3] + CE1[2, 2]
    VO[0, 1] = CE1[0, 5] + CE1[1, 5] + CE1[3, 4]
    VO[0, 2] = CE1[0, 4] + CE1[2, 4] + CE1[3, 5]
    VO[1, 2] = CE1[1, 3] + CE1[2, 3] + CE1[4, 5]
    VO = symFromUpper(VO)
    K = np.diag(DI).sum() / 9
    G = (np.diag(VO).sum() - 3 * K) / 10
    # Calculate Anisotropy
    XH = np.hstack((np.repeat(K + 4 * G / 3, 3),
                    np.repeat(np.sqrt(2) * (K - 2 * G / 3), 3),
                    np.repeat(2 * G, 3), np.repeat(0, 12)))
    ANIS = norm(XEC - XH, ord=2)
    # Swap first and third column (to sort eigenvectors in descending
    # eigenvalues order)
    eigvecDI = eigh(DI)[1][:, ::-1]
    eigvecVO = eigh(VO)[1][:, ::-1]
    # Search for SCCA directions
    for i in range(3):
        NDVC = 0
        ADVC = 10
        for j in range(3):
            SDV = np.clip(np.dot(np.ascontiguousarray(eigvecDI[:, i]),
                                 np.ascontiguousarray(eigvecVO[:, j])), -1, 1)
            ADV = np.arccos(np.absolute(SDV))
            if ADV < ADVC:
                NDVC = (j + 1) * np.sign(SDV) if SDV != 0 else j + 1
                ADVC = ADV
        eigvecDI[:, i] = (eigvecDI[:, i] + NDVC
                          * eigvecVO[:, int(abs(NDVC) - 1)]) / 2
        eigvecDI[:, i] /= norm(eigvecDI[:, i], ord=2)
    # Higher symmetry axis
    SCC = np.transpose(eigvecDI)
    for i in range(3):
        IHS = [(i + j) % 3 for j in range(3)]
        for j in range(3):
            eigvecDI[j, :] = SCC[IHS[j], :]
        DEV = trIsoProj(mat2vec(tens2mat(rotateTens(EL1, eigvecDI))))
        if DEV < XN:
            XN = DEV
            NDVC = i + 1
    eigvecDI = SCC
    IHS = [int(abs(NDVC) - 1 + i) % 3 for i in range(3)]
    for i in range(3):
        SCC[i, :] = eigvecDI[IHS[i], :]
    # Rotate in SCCA
    XEC = mat2vec(tens2mat(rotateTens(EL1, SCC)))
    return ANIS, SCC, XEC

# Forms symmetric Cartesian system
def cal_azimuth(CE1):
    # Define azimuth and inclination of ray path
    azi=0
    inc=-90*(np.pi/180)

    # Create the cartesian vector
    Xr = np.zeros(3)
    Xr = (np.cos(azi)*np.cos(inc), -np.sin(azi)*np.cos(inc), np.sin(inc))
    r = np.sqrt(Xr[0]**2+Xr[1]**2+Xr[2]**2)
    Xr = Xr/r

    # Compute Eigenvector
    gamma = np.zeros((3,6))
    gamma[0,0] = gamma[1,5] = gamma[2,4] = Xr[0]
    gamma[0,5] = gamma[1,1] = gamma[2,3] = Xr[1]
    gamma[0,4] = gamma[1,3] = gamma[2,2] = Xr[2]

    T1 = np.dot(gamma,CE1)
    T = np.dot(T1,np.transpose(gamma))
    eigval, eigvec = eigh(T)

    S1 = eigvec[:,1]

    ##calculate projection onto propagation plane
    S1N = np.zeros(3)
    S1P = np.zeros(3)
    S1N = np.cross(Xr,S1)
    S1P = np.cross(Xr,S1N)

    ##rotate into y-z plane to calculate angles
    RR = np.zeros((3,3))
    RR[0,:] = [np.cos(azi), np.sin(azi), 0]
    RR[1,:] = [-np.sin(azi), np.cos(azi), 0]
    RR[2,:] = [0, 0, 1]
    VR = np.dot(S1P,RR)

    RR2 = np.zeros((3,3))
    RR2[0,:] = [np.cos(inc), 0, -np.sin(inc)]
    RR2[1,:] = [0, 1, 0]
    RR2[2,:] = [np.sin(inc), 0, np.cos(inc)]
    VR2 = np.dot(VR,RR2)

    ph = np.arctan2(VR2[1],VR2[2])*180/np.pi

    #transform angle between -90 and 90
    if (ph < -90):
        ph = ph + 180
    elif (ph > 90):
        ph = ph -180
    return ph


# Decomposition into transverse isotropy tensor
def decsym(Sav):
    CE1 = symFromUpper(Sav)
    toprad = (CE1[0, 0] + CE1[1, 1]) / 8 - CE1[0, 1] / 4 + CE1[5, 5] / 2
    botrad = (CE1[3, 3] + CE1[4, 4]) / 2
    EPSRAD = toprad / botrad
    perc1 = (CE1[4, 4] - CE1[3, 3]) / 2
    perc2 = CE1[4, 3]
    EPSPERC = np.sqrt(perc1 ** 2 + perc2 ** 2)
    AZIMUTH = cal_azimuth(CE1)
    EL1 = mat2tens(CE1)
    XEC = mat2vec(CE1)
    XN = norm(XEC, ord=2)
    ANIS, SCC, XEC = scca(CE1, EL1, XN, XEC)
    DC5 = trIsoProj(XEC)
    PERC = (ANIS - DC5) / XN * 100
    TIAXIS = SCC[2, :]
    TIAXIS /= norm(TIAXIS, ord=2)
    INCLTI = np.arcsin(TIAXIS[2])
    return PERC, INCLTI, EPSRAD, EPSPERC, AZIMUTH


# Interpolates the velocity vector at a given point
def interpVel(pointCoords, dictGlobals=None):
    ''' Uncomment this block to use Ray
    if dictGlobals:
        iVelX = dictGlobals['iVelX']
        iVelZ = dictGlobals['iVelZ']
        if len(pointCoords) == 3:
            iVelY = dictGlobals['iVelY']
    '''
    pointVel = np.array(
        [iVelX(*pointCoords),
         (iVelY(*pointCoords) if len(pointCoords) == 3 else np.nan),
         -iVelZ(*pointCoords)])
    return pointVel[~np.isnan(pointVel)]


# Interpolates the velocity gradient tensor at a given point
def interpVelGrad(pointCoords, dictGlobals=None, ivpIter=False):
    ''' Uncomment this block to use Ray
    if dictGlobals:
        iLxx = dictGlobals['iLxx']
        iLxz = dictGlobals['iLxz']
        iLzx = dictGlobals['iLzx']
        iLzz = dictGlobals['iLzz']
        if len(pointCoords) == 3:
            iLxy = dictGlobals['iLxy']
            iLyx = dictGlobals['iLyx']
            iLyy = dictGlobals['iLyy']
            iLyz = dictGlobals['iLyz']
            iLzy = dictGlobals['iLzy']
    '''
    if len(pointCoords) == 2:
        L = np.array([[iLxx(*pointCoords), 0, -iLxz(*pointCoords)],
                      [0, 0, 0],
                      [-iLzx(*pointCoords), 0, iLzz(*pointCoords)]])
    else:
        L = np.array(
            [np.hstack(
                (iLxx(pointCoords), iLxy(pointCoords), -iLxz(pointCoords))),
             np.hstack(
                (iLyx(pointCoords), iLyy(pointCoords), -iLyz(pointCoords))),
             np.hstack(
                (-iLzx(pointCoords), -iLzy(pointCoords), iLzz(pointCoords)))])
    assert abs(np.sum(np.diag(L))) < 1e-15
    if ivpIter:
        return L
    e = (L + np.transpose(L)) / 2  # strain rate tensor
    epsnot = eigvalsh(e).__abs__().max()  # reference strain rate
    return L, epsnot


# Rotation of a matrix line based on finite strain ellipsoid
@jit(nopython=True)
def fseDecomp(fse, ex):
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
    g = np.zeros((3, 3))
    g_ens = np.zeros((3, 3))
    rt = np.zeros(size)
    dotacs = np.zeros((size, 3, 3))
    rt_ens = np.zeros(size)
    dotacs_ens = np.zeros((size, 3, 3))
    if alpha == 0:
        rotIni = fseDecomp(fse, ex)
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
def strain(pathTime, pathDense, dictGlobals=None):
    ''' Uncomment this block to use Ray
    if dictGlobals:
        acs0 = dictGlobals['acs0']
        chi = dictGlobals['chi']
        gridCoords = dictGlobals['gridCoords']
        iDefMech = dictGlobals['iDefMech']
        size = dictGlobals['size']
    '''
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
    C0 = np.zeros((3, 3, 3, 3))
    C03 = np.zeros((3, 3, 3, 3))
    Cav = np.zeros((3, 3, 3, 3))
    Sav = np.zeros((6, 6))
    # Elastic tensors c0_{ijkl}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ll in range(3):
                    C0[i, j, k, ll] = S0[ijkl[i, j], ijkl[k, ll]]
                    C03[i, j, k, ll] = S0_ens[ijkl[i, j], ijkl[k, ll]]
    for nu in range(size):
        Cav2 = np.zeros((3, 3, 3, 3))
        Cav3 = np.zeros((3, 3, 3, 3))
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
    # Average stiffness matrix
    for i in range(6):
        for j in range(6):
            Sav[i, j] = Cav[l1[i], l2[i], l1[j], l2[j]]
    return Sav


# Calculates GOL parameter at grid point
def pipar(currPoint, dictGlobals=None):
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
                a = (ind + 1) % 3
                b = (ind + 2) % 3
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
                        method='RK45', first_step=1e11, max_step=np.inf,
                        t_eval=None, events=[maxStrain], args=(dictGlobals,),
                        dense_output=True, jac=ivpJac, atol=1e-8, rtol=1e-5)
    return sol.t, sol.sol


# Checks if a point lies within the numerical domain
def isInside(point, dictGlobals=None):
    ''' Uncomment this block to use Ray
    if dictGlobals:
        gridMax = dictGlobals['gridMax']
        gridMin = dictGlobals['gridMin']
    '''
    inside = True
    for coord, minBound, maxBound in zip(point, gridMin, gridMax):
        if coord < minBound or coord > maxBound:
            inside = False
            break
    return inside


def DRex(locInd, dictGlobals=None):
    ''' Uncomment this block to use Ray
    if dictGlobals:
        gridCoords = dictGlobals['gridCoords']
    '''
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
    eigval, eigvects = eigh(LSij)
    # pick up the orientation of the long axis of the FSE
    phi_fse = np.arctan2(eigvects[-1, -1], np.sqrt(eigvects[0, -1] ** 2 + eigvects[1, -1] ** 2))
    # natural strain = ln(a / c) where a is the long axis = max(eigval) ** 0.5
    ln_fse = np.log(eigval[-1] / eigval[0]) / 2
    # Cijkl tensor (using Voigt average)
    Sav = voigt(acs, acs_ens, odf, odf_ens)
    # percentage anisotropy and orientation of hexagonal symmetry axis
    perc_a, phi_a, radani, percani, azi_direct = decsym(Sav)
    return locInd[::-1], radani, percani, GOL, ln_fse, phi_fse, phi_a, perc_a, azi_direct


def main(inputArgs):
    if isinstance(inputArgs, list):
        inputArgs = args
    elif inputArgs.ray:
        global alt, ijkl, l1, l2
    else:
        global acs0, alt, iDefMech, iVelX, iVelY, iVelZ, iLxx, iLyx, iLzx, \
            iLxy, iLyy, iLzy, iLxz, iLyz, iLzz, ijkl, l1, l2

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

    if dim == 2:
        iDefMech = NearestNDInterpolator(coords[:, :-1], defMech)
        if inputArgs.mpl:
            triIDs = []
            for i in range(vtkOut.GetNumberOfCells()):
                assert vtkOut.GetCell(i).GetNumberOfPoints() == 3
                IDsList = vtkOut.GetCell(i).GetPointIds()
                triIDs.append([IDsList.GetId(j)
                               for j in range(IDsList.GetNumberOfIds())])
            tri = Triangulation(coords[:, 0], coords[:, 1], triangles=triIDs)
            iVelX = CubicTriInterpolator(tri, vel[:, 0], kind=inputArgs.mpl)
            iVelZ = CubicTriInterpolator(tri, vel[:, 1], kind=inputArgs.mpl)
            iLxx = CubicTriInterpolator(tri, velGrad[:, 0], kind=inputArgs.mpl)
            iLzx = CubicTriInterpolator(tri, velGrad[:, 1], kind=inputArgs.mpl)
            iLxz = CubicTriInterpolator(tri, velGrad[:, 3], kind=inputArgs.mpl)
            iLzz = CubicTriInterpolator(tri, velGrad[:, 4], kind=inputArgs.mpl)
        else:
            tri = Delaunay(coords[:, :-1])
            iVelX = CloughTocher2DInterpolator(tri, vel[:, 0])
            iVelZ = CloughTocher2DInterpolator(tri, vel[:, 1])
            iLxx = CloughTocher2DInterpolator(tri, velGrad[:, 0])
            iLzx = CloughTocher2DInterpolator(tri, velGrad[:, 1])
            iLxz = CloughTocher2DInterpolator(tri, velGrad[:, 3])
            iLzz = CloughTocher2DInterpolator(tri, velGrad[:, 4])
        if inputArgs.charm:
            charm.thisProxy.updateGlobals(
                {'iDefMech': iDefMech, 'iVelX': iVelX, 'iVelZ': iVelZ,
                 'iLxx': iLxx, 'iLzx': iLzx, 'iLxz': iLxz, 'iLzz': iLzz},
                awaitable=True).get()
    else:
        iDefMech = NearestNDInterpolator(coords, defMech)
        iVelX = NearestNDInterpolator(coords, vel[:, 0])
        iVelY = NearestNDInterpolator(coords, vel[:, 1])
        iVelZ = NearestNDInterpolator(coords, vel[:, 2])
        iLxx = NearestNDInterpolator(coords, velGrad[:, 0])
        iLyx = NearestNDInterpolator(coords, velGrad[:, 1])
        iLzx = NearestNDInterpolator(coords, velGrad[:, 2])
        iLxy = NearestNDInterpolator(coords, velGrad[:, 3])
        iLyy = NearestNDInterpolator(coords, velGrad[:, 4])
        iLzy = NearestNDInterpolator(coords, velGrad[:, 5])
        iLxz = NearestNDInterpolator(coords, velGrad[:, 6])
        iLyz = NearestNDInterpolator(coords, velGrad[:, 7])
        iLzz = NearestNDInterpolator(coords, velGrad[:, 8])
        if inputArgs.charm:
            charm.thisProxy.updateGlobals(
                {'iDefMech': iDefMech, 'iVelX': iVelX, 'iVelY': iVelY,
                 'iVelZ': iVelZ, 'iLxx': iLxx, 'iLyx': iLyx, 'iLzx': iLzx,
                 'iLxy': iLxy, 'iLyy': iLyy, 'iLzy': iLzy, 'iLxz': iLxz,
                 'iLyz': iLyz, 'iLzz': iLzz},
                awaitable=True).get()
    del vtkOut

    alt = np.zeros((3, 3, 3))  # \epsilon_{ijk}
    for ii in range(3):
        alt[ii % 3, (ii + 1) % 3, (ii + 2) % 3] = 1
        alt[ii % 3, (ii + 2) % 3, (ii + 1) % 3] = -1
    # Indices to form Cijkl from Sij
    ijkl = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype='int')
    # Indices to form Sij from Cijkl
    l1 = np.array([0, 1, 2, 1, 2, 0], dtype='int')
    l2 = np.array([0, 1, 2, 2, 0, 1], dtype='int')
    # Direction cosine matrix with uniformly distributed rotations.
    acs0 = R.random(size, random_state=1).as_matrix()

    if inputArgs.restart:
        checkpointVar = np.load(inputArgs.restart)
        GOL = checkpointVar['GOL']
        phi_fse = checkpointVar['phi_fse']
        ln_fse = checkpointVar['ln_fse']
        perc_a = checkpointVar['perc_a']
        phi_a = checkpointVar['phi_a']
        radani = checkpointVar['radani']
        percani = checkpointVar['percani']
        indArr = checkpointVar['indArr']
        nodesComplete = checkpointVar['nodesComplete']
    else:
        arrDim = [x - 1 for x in reversed(gridNodes)]
        GOL = np.zeros(arrDim)
        phi_fse = np.zeros(arrDim)
        ln_fse = np.zeros(arrDim)
        perc_a = np.zeros(arrDim)
        phi_a = np.zeros(arrDim)
        radani = np.zeros(arrDim)
        percani = np.zeros(arrDim)
        azi_direct = np.zeros(arrDim)
        indArr = np.zeros(arrDim[::-1])
        nodesComplete = 0

    if inputArgs.charm:
        # Charm with Checkpoint
        charm.thisProxy.updateGlobals(
            {'chi': chi, 'gridMin': gridMin, 'gridMax': gridMax, 'lamb': lamb,
             'mob': mob, 'size': size, 'stressexp': stressexp, 'S0': S0,
             'S0_ens': S0_ens, 'tau': tau, 'tau_ens': tau_ens, 'Xol': Xol,
             'gridNodes': gridNodes, 'gridCoords': gridCoords, 'alt': alt,
             'ijkl': ijkl, 'l1': l1, 'l2': l2, 'acs0': acs0},
            awaitable=True).get()

        for batch in range(np.ceil(np.sum(indArr == 0) / 6e4).astype(int)):
            nodes2do = np.asarray(indArr == 0).nonzero()
            futures = charm.pool.map_async(
                DRex, list(zip(*[nodes[:int(6e4)] for nodes in nodes2do])),
                multi_future=True)
            for future in charm.iwait(futures):
                for r0, r1, r2, r3, r4, r5, r6, r7, r8 in [future.get()]:
                    radani[r0] = r1
                    percani[r0] = r2
                    GOL[r0] = r3
                    ln_fse[r0] = r4
                    phi_fse[r0] = r5
                    phi_a[r0] = r6
                    perc_a[r0] = r7
                    azi_direct[r0] = r8
                    indArr[r0[::-1]] = 1
                    nodesComplete += 1
                if not nodesComplete % checkpoint:
                    np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                             f'{nodesComplete}',
                             radani=radani, percani=percani, GOL=GOL,
                             ln_fse=ln_fse, phi_fse=phi_fse, phi_a=phi_a,
                             perc_a=perc_a, azi_direct=azi_direct,indArr=indArr,
                             nodesComplete=nodesComplete)
    elif inputArgs.ray:
        # Ray with Checkpoint
        if inputArgs.redis_pass:
            # Cluster execution
            ray.init(address='auto', redis_password=inputArgs.redis_pass)
        else:
            # Single machine | Set local_mode to True to force serial execution
            ray.init(num_cpus=inputArgs.cpus, local_mode=False)

        if dim == 2:
            dictGlobals = {
                'iDefMech': iDefMech, 'iVelX': iVelX, 'iVelZ': iVelZ,
                'iLxx': iLxx, 'iLzx': iLzx, 'iLxz': iLxz, 'iLzz': iLzz,
                'chi': chi, 'gridMin': gridMin, 'gridMax': gridMax,
                'size': size, 'gridNodes': gridNodes, 'gridCoords': gridCoords,
                'acs0': acs0}
        else:
            dictGlobals = {
                'iDefMech': iDefMech, 'iVelX': iVelX, 'iVelY': iVelY,
                'iVelZ': iVelZ, 'iLxx': iLxx, 'iLyx': iLyx, 'iLzx': iLzx,
                'iLxy': iLxy, 'iLyy': iLyy, 'iLzy': iLzy, 'iLxz': iLxz,
                'iLyz': iLyz, 'iLzz': iLzz, 'chi': chi, 'gridMin': gridMin,
                'gridMax': gridMax, 'size': size, 'gridNodes': gridNodes,
                'gridCoords': gridCoords, 'acs0': acs0}
        dictGlobalsID = ray.put(dictGlobals)

        for batch in range(np.ceil(np.sum(indArr == 0) / 6e4).astype(int)):
            nodes2do = np.asarray(indArr == 0).nonzero()
            futures = [DRex.remote(i, dictGlobalsID)
                       for i in zip(*[nodes[:int(6e4)] for nodes in nodes2do])]
            while len(futures) > 0:
                readyId, remainingIds = ray.wait(
                    futures, num_returns=min([checkpoint, len(futures)]))
                for r0, r1, r2, r3, r4, r5, r6, r7, r8 in ray.get(readyId):
                    radani[r0] = r1
                    percani[r0] = r2
                    GOL[r0] = r3
                    ln_fse[r0] = r4
                    phi_fse[r0] = r5
                    phi_a[r0] = r6
                    perc_a[r0] = r7
                    azi_direct[r0] = r8
                    indArr[r0[::-1]] = 1
                    nodesComplete += 1
                np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                         f'{nodesComplete}',
                         radani=radani, percani=percani, GOL=GOL,
                         ln_fse=ln_fse, phi_fse=phi_fse, phi_a=phi_a,
                         perc_a=perc_a, azi_direct=azi_direct, indArr=indArr,
                         nodesComplete=nodesComplete)
                futures = remainingIds

        ray.shutdown()
    else:
        if __name__ == '__main__':
            # Multiprocessing with Checkpoint
            with Pool(processes=inputArgs.cpus) as pool:
                for r0, r1, r2, r3, r4, r5, r6, r7, r8 in pool.imap_unordered(
                        DRex, zip(*np.asarray(indArr == 0).nonzero())):
                    radani[r0] = r1
                    percani[r0] = r2
                    GOL[r0] = r3
                    ln_fse[r0] = r4
                    phi_fse[r0] = r5
                    phi_a[r0] = r6
                    perc_a[r0] = r7
                    azi_direct[r0] = r8
                    indArr[r0[::-1]] = 1
                    nodesComplete += 1
                    if not nodesComplete % checkpoint:
                        np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_'
                                 f'{nodesComplete}',
                                 radani=radani, percani=percani, GOL=GOL,
                                 ln_fse=ln_fse, phi_fse=phi_fse, phi_a=phi_a,
                                 perc_a=perc_a, azi_direct=azi_direct,
                                 indArr=indArr, nodesComplete=nodesComplete)

    np.savez(f'PyDRex{dim}D_{name}_Complete', radani=radani, percani=percani,
             GOL=GOL, ln_fse=ln_fse, phi_fse=phi_fse, phi_a=phi_a,
             perc_a=perc_a, azi_direct=azi_direct)

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
    from charm4py import charm  # noqa: F401
elif args.ray:
    import ray
    DRex = ray.remote(DRex)
else:
    from multiprocessing import Pool

if args.mpl:
    from matplotlib.tri import (CubicTriInterpolator,  # noqa: F401
                                Triangulation)
else:
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.spatial import Delaunay

if args.charm:
    charm.start(main)
else:
    main(args)
