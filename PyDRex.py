#!/usr/bin/env python3

###############################################################################
# Python 3 Version of D-Rex
###############################################################################

import argparse
from numba import jit
import numpy as np
from numpy.linalg import eigh, eigvalsh, norm
import ray
from ray.util import ActorPool
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from time import perf_counter
import vtk

from DRexParam import checkpoint, dim, gridNodes, name

###############################################################################
#
# symFromUpper  : Extracts the upper triangle part of an array and uses it to
#                 make a symmetric array
# mat2tens      : Yields the 4th-order tensor equivalent of a 6x6 matrix
# tens2mat      : Yields the 6x6 matrix equivalent of a 4th-order tensor
# mat2vec       : Yields the 21 components vector equivalent of a 6x6 matrix
# rotateTens    : Rotates a 4th-order tensor
# PROJECTI      : Transverse isotropy projector
# SCCA          : Forms symmetric cartesian system
# DECSYM        : Decomposition into transverse isotropy tensor
# interpVel     : Interpolates the velocity vector at a given point
# interpVelGrad : Interpolates the velocity gradient tensor at a given point
# fseDecomp     : Rotates a matrix line based on finite strain ellipsoid
# DERIV         : Calculation of the rotation vector and slip rate
# strain        : Calculation of strain along pathlines
# # EIGEN       : Find 3 eigen values of velocity gradient tensor
# VOIGT         : Calculates elastic tensor cav_{ijkl} for olivine
# PIPAR         : Calculates GOL parameter at grid point
# ISACALC       : Calculates ISA Orientation at grid point
# pathline      : Calculates pathline for individual particle
# isInside      : Checks if coordinates map to within the domain
#
###############################################################################


@ray.remote
class DRexClass(object):
    def __init__(self, inputArgs):
        self.param = __import__('DRexParam')
        inputExt = inputArgs.input.split('.')[-1]
        if inputExt == 'vtu':
            vtkReader = vtk.vtkXMLUnstructuredGridReader()
        elif inputExt == 'pvtu':
            vtkReader = vtk.vtkXMLPUnstructuredGridReader()
        else:
            raise RuntimeError(
                '''Please either provide a supported file format or implement
                the reader your require.''') from None
        vtkReader.SetFileName(inputArgs.input)
        vtkReader.Update()
        vtkOut = vtkReader.GetOutput()
        vtkData = vtkOut.GetPoints().GetData()
        coords = np.array([vtkData.GetTuple3(tup)
                           for tup in range(vtkData.GetNumberOfTuples())])
        vtkScalars = vtkOut.GetPointData().GetScalars('DeformationMechanism')
        defMech = np.array([vtkScalars.GetTuple1(tup)
                            for tup in range(vtkScalars.GetNumberOfTuples())])
        vtkScalars = vtkOut.GetPointData().GetScalars('Velocity')
        vel = np.array([vtkScalars.GetTuple3(tup)
                        for tup in range(vtkScalars.GetNumberOfTuples())])
        vtkArray = vtkOut.GetPointData().GetArray('VelocityGradient')
        velGrad = np.array([vtkArray.GetTuple9(tup)
                            for tup in range(vtkArray.GetNumberOfTuples())])
        del vtkReader, vtkOut, vtkData, vtkScalars, vtkArray
        if self.param.dim == 2:
            coords[:, 1] = np.amax(coords[:, 1]) - coords[:, 1]
            self.iDefMech = NearestNDInterpolator(coords[:, :-1], defMech)
            tri = Delaunay(coords[:, :-1])
            self.iVelX = CloughTocher2DInterpolator(tri, vel[:, 0])
            self.iVelZ = CloughTocher2DInterpolator(tri, vel[:, 1])
            self.iLxx = CloughTocher2DInterpolator(tri, velGrad[:, 0])
            self.iLzx = CloughTocher2DInterpolator(tri, velGrad[:, 1])
            self.iLxz = CloughTocher2DInterpolator(tri, velGrad[:, 3])
            self.iLzz = CloughTocher2DInterpolator(tri, velGrad[:, 4])
            del tri
        else:
            coords[:, 2] = np.amax(coords[:, 2]) - coords[:, 2]
            self.iDefMech = NearestNDInterpolator(coords, defMech)
            self.iVelX = NearestNDInterpolator(coords, vel[:, 0])
            self.iVelY = NearestNDInterpolator(coords, vel[:, 1])
            self.iVelZ = NearestNDInterpolator(coords, vel[:, 2])
            self.iLxx = NearestNDInterpolator(coords, velGrad[:, 0])
            self.iLyx = NearestNDInterpolator(coords, velGrad[:, 1])
            self.iLzx = NearestNDInterpolator(coords, velGrad[:, 2])
            self.iLxy = NearestNDInterpolator(coords, velGrad[:, 3])
            self.iLyy = NearestNDInterpolator(coords, velGrad[:, 4])
            self.iLzy = NearestNDInterpolator(coords, velGrad[:, 5])
            self.iLxz = NearestNDInterpolator(coords, velGrad[:, 6])
            self.iLyz = NearestNDInterpolator(coords, velGrad[:, 7])
            self.iLzz = NearestNDInterpolator(coords, velGrad[:, 8])
        self.alt = np.zeros((3, 3, 3))  # \epsilon_{ijk}
        for ii in range(3):
            self.alt[ii % 3, (ii + 1) % 3, (ii + 2) % 3] = 1
            self.alt[ii % 3, (ii + 2) % 3, (ii + 1) % 3] = -1
        # ijkl -> Tensor of indices to form Cijkl from Sij
        self.ijkl = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype='int')
        # l1, l2 -> Tensors of indices to form Sij from Cijkl
        self.l1 = np.array([0, 1, 2, 1, 2, 0], dtype='int')
        self.l2 = np.array([0, 1, 2, 2, 0, 1], dtype='int')
        self.acs0 = R.random(self.param.size, random_state=1).as_matrix()

    # Extracts the upper triangle part of an array and uses it to make a
    # symmetric array
    @staticmethod
    def symFromUpper(arr):
        return (np.triu(arr) + np.transpose(np.triu(arr))
                - np.diag(np.diag(arr)))

    # Yields the 4th-order tensor equivalent of a 6x6 matrix
    @staticmethod
    def mat2tens(mat):
        tens = np.zeros((3, 3, 3, 3))
        for ii in range(3):
            for jj in range(3):
                deltaIJ = 1 if ii == jj else 0
                p = deltaIJ * (ii + 1) + (1 - deltaIJ) * (7 - ii - jj) - 1
                for kk in range(3):
                    for ll in range(3):
                        deltaKL = 1 if kk == ll else 0
                        q = (deltaKL * (kk + 1)
                             + (1 - deltaKL) * (7 - kk - ll) - 1)
                        tens[ii, jj, kk, ll] = mat[p, q]
        return tens

    # Yields the 6x6 matrix equivalent of a 4th-order tensor
    @staticmethod
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
        return mat + np.transpose(mat) - np.diag(np.diag(mat))

    # Yields the 21 components vector equivalent of a 6x6 matrix
    @staticmethod
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
    @staticmethod
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

    # PROJECTI: Transverse isotropy projector
    @staticmethod
    def PROJECTI(X):
        XH = np.zeros(21)
        XH[0] = 3 / 8 * (X[0] + X[1]) + X[5] / 4 / np.sqrt(2) + X[8] / 4
        XH[1] = 3 / 8 * (X[0] + X[1]) + X[5] / 4 / np.sqrt(2) + X[8] / 4
        XH[2] = X[2]
        XH[3] = XH[4] = (X[3] + X[4]) / 2
        XH[5] = ((X[0] + X[1]) / 4 / np.sqrt(2) + 3 / 4 * X[5] - X[8] / 2 /
                 np.sqrt(2))
        XH[6] = XH[7] = (X[6] + X[7]) / 2
        XH[8] = (X[0] + X[1]) / 4 - X[5] / 2 / np.sqrt(2) + X[8] / 2
        return norm(X - XH, ord=2)

    # SCCA: forms symmetric cartesian system
    def SCCA(self, CE1, EL1, XN, XEC):
        DI = np.zeros((3, 3))
        VO = np.zeros((3, 3))
        for i in range(3):
            DI[i, i] = np.sum(CE1[i, :3])
        DI[0, 1] = np.sum(CE1[5, :3])
        DI[0, 2] = np.sum(CE1[4, :3])
        DI[1, 2] = np.sum(CE1[3, :3])
        DI = self.symFromUpper(DI)
        VO[0, 0] = CE1[0, 0] + CE1[5, 5] + CE1[4, 4]
        VO[1, 1] = CE1[5, 5] + CE1[1, 1] + CE1[3, 3]
        VO[2, 2] = CE1[4, 4] + CE1[3, 3] + CE1[2, 2]
        VO[0, 1] = CE1[0, 5] + CE1[1, 5] + CE1[3, 4]
        VO[0, 2] = CE1[0, 4] + CE1[2, 4] + CE1[3, 5]
        VO[1, 2] = CE1[1, 3] + CE1[2, 3] + CE1[4, 5]
        VO = self.symFromUpper(VO)
        K = np.sum(np.diag(DI)) / 9
        G = (np.sum(np.diag(VO)) - 3 * K) / 10
        # Calculate Anisotropy
        XH = np.hstack((np.repeat(K + 4 * G / 3, 3),
                        np.repeat(np.sqrt(2) * (K - 2 * G / 3), 3),
                        np.repeat(2 * G, 3), np.repeat(0, 12)))
        ANIS = norm(XEC - XH, ord=2)
        # Swap first and third column (to sort eigenvectors in descending
        # eigenvalues order)
        eigvecDI = np.ascontiguousarray(eigh(DI)[1][:, ::-1])
        eigvecVO = np.ascontiguousarray(eigh(VO)[1][:, ::-1])
        # Search for SCCA directions
        for i in range(3):
            NDVC = 0
            ADVC = 10
            for j in range(3):
                SDV = np.clip(np.dot(eigvecDI[:, i], eigvecVO[:, j]), -1, 1)
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
            DEV = self.PROJECTI(self.mat2vec(self.tens2mat(self.rotateTens(
                EL1, eigvecDI))))
            if DEV < XN:
                XN = DEV
                NDVC = i + 1
        eigvecDI = SCC
        IHS = [int(abs(NDVC) - 1 + i) % 3 for i in range(3)]
        for i in range(3):
            SCC[i, :] = eigvecDI[IHS[i], :]
        # Rotate in SCCA
        XEC = self.mat2vec(self.tens2mat(self.rotateTens(EL1, SCC)))
        return ANIS, SCC, XEC

    # DECSYM: Decomposition into transverse isotropy tensor
    def DECSYM(self, Sav):
        CE1 = self.symFromUpper(Sav)
        toprad = (CE1[0, 0] + CE1[1, 1]) / 8 - CE1[0, 1] / 4 + CE1[5, 5] / 2
        botrad = (CE1[3, 3] + CE1[4, 4]) / 2
        EPSRAD = toprad / botrad
        perc1 = (CE1[4, 4] - CE1[3, 3]) / 2
        perc2 = CE1[4, 3]
        EPSPERC = np.sqrt(perc1 ** 2 + perc2 ** 2)
        EL1 = self.mat2tens(CE1)
        XEC = self.mat2vec(CE1)
        XN = norm(XEC, ord=2)
        ANIS, SCC, XEC = self.SCCA(CE1, EL1, XN, XEC)
        DC5 = self.PROJECTI(XEC)
        PERC = (ANIS - DC5) / XN * 100
        TIAXIS = SCC[2, :]
        TIAXIS /= norm(TIAXIS, ord=2)
        INCLTI = np.arcsin(TIAXIS[2])
        return PERC, INCLTI, EPSRAD, EPSPERC

    # Interpolates the velocity vector at a given point
    def interpVel(self, pointCoords):
        pointVel = np.array(
            [self.iVelX(*pointCoords),
             self.iVelY(*pointCoords) if len(pointCoords) == 3 else np.nan,
             -self.iVelZ(*pointCoords)])
        return pointVel[~np.isnan(pointVel)]

    # Interpolates the velocity gradient tensor at a given point
    def interpVelGrad(self, pointCoords):
        if len(pointCoords) == 2:
            L = np.array(
                [[self.iLxx(*pointCoords), 0, -self.iLxz(*pointCoords)],
                 [0, 0, 0],
                 [-self.iLzx(*pointCoords), 0, self.iLzz(*pointCoords)]])
        else:
            L = np.array(
                [np.hstack(
                    (self.iLxx(pointCoords), self.iLxy(pointCoords),
                     -self.iLxz(pointCoords))),
                 np.hstack(
                    (self.iLyx(pointCoords), self.iLyy(pointCoords),
                     -self.iLyz(pointCoords))),
                 np.hstack(
                    (-self.iLzx(pointCoords), -self.iLzy(pointCoords),
                     self.iLzz(pointCoords)))])
        assert abs(np.sum(np.diag(L))) < 1e-15
        e = (L + np.transpose(L)) / 2
        epsnot = np.amax(np.absolute(eigvalsh(e)))
        return L, epsnot

    def DERIV(self, lx, ex, acsi, acsi_ens, fse, odfi, odfi_ens, alpha):
        return DERIV_jit(lx, ex, acsi, acsi_ens, fse, odfi, odfi_ens, alpha,
                         self.param.size, self.param.tau, self.param.tau_ens,
                         self.param.stressexp, self.param.mob, self.alt,
                         self.param.Xol, self.param.lamb)

    # strain: Calculation of strain along pathlines
    def strain(self, step, stream_Lij, stream_e, stream_dt, stream_alpha):
        fse = np.identity(3)
        odf = 1 / self.param.size * np.ones(self.param.size)
        odf_ens = 1 / self.param.size * np.ones(self.param.size)
        acs = np.copy(self.acs0)
        acs_ens = np.copy(self.acs0)
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
                dotacs, dotacs_ens, dotodf, dotodf_ens = self.DERIV(
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
                odfi /= np.sum(odfi)
                odfi_ens = np.clip(odf_ens + 0.5 * kodf1_ens, 0, None)
                odfi_ens /= np.sum(odfi_ens)
                # CALL 2/4
                dotacs, dotacs_ens, dotodf, dotodf_ens = self.DERIV(
                    L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi,
                    odfi_ens, alpha)
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
                dotacs, dotacs_ens, dotodf, dotodf_ens = self.DERIV(
                    L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi,
                    odfi_ens, alpha)
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
                dotacs, dotacs_ens, dotodf, dotodf_ens = self.DERIV(
                    L / epsnot, e / epsnot, acsi, acsi_ens, fse, odfi,
                    odfi_ens, alpha)
                kfse4 = np.dot(L, fsei) * dt
                kodf4 = dotodf * dt * epsnot
                kodf4_ens = dotodf_ens * dt * epsnot
                kac4 = dotacs * dt * epsnot
                kac4_ens = dotacs_ens * dt * epsnot
                fse += (kfse1 / 2 + kfse2 + kfse3 + kfse4 / 2) / 3
                acs = np.clip(acs + (kac1 / 2 + kac2 + kac3 + kac4 / 2) / 3,
                              -1, 1)
                acs_ens = np.clip(acs_ens + (kac1_ens / 2 + kac2_ens + kac3_ens
                                             + kac4_ens / 2) / 3, -1, 1)
                odf += (kodf1 / 2 + kodf2 + kodf3 + kodf4 / 2) / 3
                odf_ens += (kodf1_ens / 2 + kodf2_ens +
                            kodf3_ens + kodf4_ens / 2) / 3
                odf /= np.sum(odf)
                odf_ens /= np.sum(odf_ens)
                # Grain-boundary sliding
                mask = odf <= self.param.chi / self.param.size
                mask_ens = odf_ens <= self.param.chi / self.param.size
                acs[mask, :, :] = self.acs0[mask, :, :]
                acs_ens[mask_ens, :, :] = self.acs0[mask_ens, :, :]
                odf[mask] = self.param.chi / self.param.size
                odf_ens[mask_ens] = self.param.chi / self.param.size
                odf /= np.sum(odf)
                odf_ens /= np.sum(odf_ens)
        return fse, odf, odf_ens, acs, acs_ens

    def VOIGT(self, odf, odf_ens, acs, acs_ens):
        S0 = self.symFromUpper(self.param.S0)
        S0_ens = self.symFromUpper(self.param.S0_ens)
        return VOIGT_jit(odf, odf_ens, acs, acs_ens, S0, S0_ens,
                         self.l1, self.l2, self.ijkl,
                         self.param.size, self.param.Xol)

    '''
    # EIGEN: Find 3 eigen values of velocity gradient tensor
    @staticmethod
    def EIGEN(L):
        Id = np.identity(3)
        Q = 0
        for ii in range(2):
            for jj in range(ii + 1, 3):
                Q -= (L[ii, ii] * L[jj, jj] - L[ii, jj] * L[jj, ii]) / 3
        R = np.linalg.det(L) / 2
        if abs(Q) < 1e-9:
            F = -np.ones((3, 3))
        elif Q ** 3 - R ** 2 >= 0:
            theta = np.arccos(R / Q ** 1.5)
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
            xx = (np.sqrt(R ** 2 - Q ** 3) + abs(R)) ** (1 / 3)
            L1 = -(xx + Q / xx) if R == 0 else -np.sign(R) * (xx + Q / xx)
            L2 = L3 = -L1 / 2
            F = (np.dot(L - L2 * Id, L - L3 * Id) if L1 > 1e-9
                 else np.zeros((3, 3)))
        return F
    '''

    # PIPAR: Calculates GOL parameter at grid point
    def PIPAR(self, currPoint):
        # ISACALC: Calculates ISA Orientation at grid point
        def ISACALC(L, veloc):
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
            '''
            # Everything commented out below is the original Fortran
            # implementation
            F = EIGEN(L)
            print(eigh(np.dot(F, np.transpose(F))), eigh(Ua))
            if np.sum(F) == -9:
                isa = -np.ones(3)
            elif np.sum(np.absolute(F)) == 0:
                isa = np.zeros(3)
                GOL = -1
            else:
                isa = eigh(np.dot(F, np.transpose(F)))[1][:, -1]
            return isa if dim == 3 else isa[::2]
            '''
        GOL = 10
        # local velocity vector and orientation of infinite strain axis
        veloc = self.interpVel(currPoint)
        dt = 1e-2 / norm(veloc, ord=2)
        # previous point on the streamline
        prevPoint = currPoint - dt * veloc
        if self.isInside(prevPoint) is False:
            return -10
        veloc = self.interpVel(prevPoint)
        # epsnot: reference strain rate
        L, epsnot = self.interpVelGrad(prevPoint)
        L /= epsnot
        # calculation of the ISA
        isa = ISACALC(L, veloc)
        '''
        isa = (veloc / norm(veloc, ord=2)
               if np.sum(ISACALC(L)) == -3 else ISACALC(L))
        '''
        # angle between ISA and flow direction
        thetaISA = np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa))
        # next point on the streamline
        nextPoint = currPoint + dt * veloc
        if self.isInside(nextPoint) is False:
            return -10
        veloc = self.interpVel(nextPoint)
        '''
        veloc = veloc / norm(veloc, ord=2)
        '''
        L, epsnot = self.interpVelGrad(nextPoint)
        L /= epsnot
        # calculation of the ISA
        '''
        isa = veloc if np.sum(ISACALC(L)) == -3 else ISACALC(L)
        '''
        isa = ISACALC(L, veloc)
        L, epsnot = self.interpVelGrad(currPoint)
        # angle between ISA and flow direction
        thetaISA = abs(thetaISA
                       - np.arccos(np.sum(veloc / norm(veloc, ord=2) * isa)))
        if thetaISA > np.pi:
            thetaISA -= np.pi
        return min(GOL, thetaISA / 2 / dt / epsnot)

    def pathline(self, currInd, currPoint):
        stream_Lij = np.zeros((3, 3, 10000))
        stream_dt = np.zeros(10000)
        stream_e = np.zeros(10000)
        stream_alpha = np.zeros(10000)
        max_strain = 0
        step = -1
        while max_strain < 10 and step < 1e4:
            step += 1
            currVel = self.interpVel(currPoint)
            L, epsnot = self.interpVelGrad(currPoint)
            dt = min([abs(self.param.gridCoords[x][currInd[x] + 1]
                      - self.param.gridCoords[x][currInd[x]])
                      for x in range(len(currInd))]) / 4 / norm(currVel, ord=2)
            stream_Lij[:, :, step] = L
            stream_dt[step] = dt
            stream_e[step] = epsnot
            stream_alpha[step] = self.iDefMech(*currPoint)
            max_strain += dt * epsnot
            k1 = -currVel * dt
            newPoint = currPoint + 0.5 * k1
            if self.isInside(newPoint) is False:
                break
            currVel = self.interpVel(newPoint)
            k2 = -currVel * dt
            newPoint = currPoint + 0.5 * k2
            if self.isInside(newPoint) is False:
                break
            currVel = self.interpVel(newPoint)
            k3 = -currVel * dt
            newPoint = currPoint + k3
            if self.isInside(newPoint) is False:
                break
            currVel = self.interpVel(newPoint)
            currPoint += (k1 / 2 + k2 + k3 - currVel * dt / 2) / 3
            if self.isInside(currPoint) is False:
                break
            for i in range(len(currPoint)):
                while (self.param.gridCoords[i][currInd[i]] > currPoint[i]
                       and currInd[i] > 0):
                    currInd[i] -= 1
                while (self.param.gridCoords[i][currInd[i] + 1] < currPoint[i]
                       and currInd[i] < self.param.gridNodes[i] - 2):
                    currInd[i] += 1
        return (step, stream_Lij[:, :, :step + 1], stream_e[:step + 1],
                stream_dt[:step + 1], stream_alpha[:step + 1])

    def isInside(self, point):
        inside = True
        for i in range(point.size):
            if (point[i] < self.param.gridMin[i]
                    or point[i] > self.param.gridMax[i]):
                inside = False
                break
        return inside

    def DRex(self, locInd):
        currCoords = np.array(
            [(self.param.gridCoords[x][locInd[x]]
              + self.param.gridCoords[x][locInd[x] + 1]) / 2
             for x in range(len(locInd))])
        # Calculation of GOL parameter
        GOL = self.PIPAR(currCoords)
        # Backward calculation of the pathline for each tracer
        step, stream_Lij, stream_e, stream_dt, stream_alpha = self.pathline(
            list(locInd), currCoords)
        # Inward calculation of the LPO
        # Random initial LPO
        Fij, odf, odf_ens, acs, acs_ens = self.strain(
            step, stream_Lij, stream_e, stream_dt, stream_alpha)
        # Left-stretch tensor for FSE calculation
        LSij = np.dot(Fij, np.transpose(Fij))
        eigval, eigvects = eigh(LSij)
        # pick up the orientation of the long axis of the FSE
        phi_fse = np.arctan2(eigvects[-1, -1], eigvects[0, -1])
        # natural strain = ln(a / c)
        # where a is the long axis = max(eigval) ** 0.5
        ln_fse = np.log(eigval[-1] / eigval[0]) / 2
        # Cijkl tensor (using Voigt average)
        Sav = self.VOIGT(odf, odf_ens, acs, acs_ens)
        # percentage anisotropy and orientation of hexagonal symmetry axis
        perc_a, phi_a, radani, percani = self.DECSYM(Sav)
        return locInd, radani, percani, ln_fse, phi_fse, phi_a, perc_a, GOL


# Rotation of a matrix line based on finite strain ellipsoid
@jit(nopython=True, nogil=True, fastmath=True)
def fseDecomp(fse, ex, alt):
    # LSij: left-strech tensor for fse calculation
    LSij = np.dot(fse, np.transpose(fse))
    eigval, eigvect = eigh(LSij)
    eigvect = np.transpose(eigvect)  # Each column of eigvect is an eigenvector
    # and its transpose is the cosines matrix
    rot = np.zeros(3)  # vector of induced spin omega_i, i=1,3
    unique = True
    for i, eigv in enumerate(eigval):
        otherEig = eigval[np.array([(i + 1) % 3, (i + 2) % 3])]
        if (np.absolute((otherEig - eigv) / eigv) < 5e-2).any():
            unique = False
            break
    if unique is False:
        return rot
    else:
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
                            int1 += (alt[ii, ll, kk]
                                     * eigvect[(nn + 2) % 3, jj]
                                     * eigvect[(nn + 2) % 3, ll])
                        int2 += ((eigvect[nn, jj] * eigvect[(nn + 1) % 3, kk]
                                  * eigvect[(nn + 2) % 3, ii] + int1)
                                 * ex[jj, kk])
                rot[ii] += H[nn] * int2
        return rot


# DERIV: Calculation of the rotation vector and slip rate
@jit(nopython=True, nogil=True, fastmath=True)
def DERIV_jit(lx, ex, acsi, acsi_ens, fse, odfi, odfi_ens, alpha,
              size, tau, tau_ens, stressexp, mob, alt, Xol, lamb):
    # lx and ex: Dimensionless velocity gradient and strain rate tensors
    # potential to update olivine fabric type in the loop from navid in future
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
    # lx += (alpha - 1) * ex  # if alpha can vary between 1 and 0
    # ex *= alpha  # would also need to add setDiffusionCreep back in
    for ii in range(size):
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
        gam0_ens = R2_ens / R1_ens * (1 / tau_ens) ** stressexp
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
            rot = np.copy(rotIni)
            rot[2] += (lx[1, 0] - lx[0, 1]) / 2
            rot[1] += (lx[0, 2] - lx[2, 0]) / 2
            rot[0] += (lx[2, 1] - lx[1, 2]) / 2
            rot_ens = np.copy(rot)
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
                        dotacs[ii, i1, i2] += (alt[i2, i3, i4] *
                                               acsi[ii, i1, i4] * rot[i3])
                        dotacs_ens[ii, i1, i2] += (alt[i2, i3, i4] *
                                                   acsi_ens[ii, i1, i4]
                                                   * rot_ens[i3])
        # Removed grain boundary sliding for small grains
        # if odfi[ii] < chi / size:
        #     dotacs[ii, :, :] = 0
        #     rt[ii] = 0
        # if odfi_ens[ii] < chi / size:
        #     dotacs_ens[ii, :, :] = 0
        #     rt_ens[ii] = 0
    # Volume averaged energy
    Emean = np.sum(odfi * rt)
    Emean_ens = np.sum(odfi_ens * rt_ens)
    # Change of volume fraction by grain boundary migration
    dotodf = Xol * mob * odfi * (Emean - rt)
    dotodf_ens = (1 - Xol) * mob * odfi_ens * (Emean_ens - rt_ens)
    return dotacs, dotacs_ens, dotodf, dotodf_ens


# VOIGT: Calculates elastic tensor cav_{ijkl} for olivine
@jit(nopython=True, nogil=True, fastmath=True)
def VOIGT_jit(odf, odf_ens, acs, acs_ens, S0, S0_ens, l1, l2, ijkl, size, Xol):
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


begin = perf_counter()

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''
Python 3 implementation of the seismic anisotropy software D-Rex. Requires the
input file DRexParam.py in the same directory.''')
parser.add_argument('input', help='input file (expects a vtkUnstructuredGrid)')
parser.add_argument('-r', '--restart', help='''checkpoint file to restart from
                    (expects a Numpy NpzFile)''', metavar='')
parser.add_argument('-m', '--multi', action='store_true',
                    help='''enables multi-node execution''')
parser.add_argument('-p', '--pw', help='''Redis password to connect to an
                    existing Ray cluster''', metavar='')
args = parser.parse_args()

try:
    checkpointVar = np.load(args.restart)
    locals().update(checkpointVar)
except TypeError:
    arrDim = [x - 1 for x in reversed(gridNodes)]
    GOL = np.zeros(arrDim)
    phi_fse = np.zeros(arrDim)
    ln_fse = np.zeros(arrDim)
    perc_a = np.zeros(arrDim)
    phi_a = np.zeros(arrDim)
    radani = np.zeros(arrDim)
    percani = np.zeros(arrDim)
    indArr = np.zeros(arrDim[::-1])
    nodesComplete = 0

if args.multi:
    # Cluster execution | Ensure the driver OSM is less than the total OSM
    ray.init(address='auto', redis_password=args.pw,
             driver_object_store_memory=10 * 1024 * 1024 * 1024)
else:
    # Single machine | Set local_mode to True to ensure serial execution
    ray.init(num_cpus=6, local_mode=False)

actors = []
for actor in range(int(ray.cluster_resources()['CPU'])):
    actors.append(DRexClass.remote(args))
pool = ActorPool(actors)
for r0, r1, r2, r3, r4, r5, r6, r7 in pool.map_unordered(
        lambda a, v: a.DRex.remote(v),
        zip(*np.asarray(indArr == 0).nonzero())):
    indArr[r0] = 1
    nodesComplete += 1
    r0 = r0[::-1]
    radani[r0] = r1
    percani[r0] = r2
    ln_fse[r0] = r3
    phi_fse[r0] = r4
    phi_a[r0] = r5
    perc_a[r0] = r6
    GOL[r0] = r7
    if not nodesComplete % checkpoint:
        np.savez(f'PyDRex{dim}D_{name}_NumpyCheckpoint_{nodesComplete}',
                 radani=radani, percani=percani, ln_fse=ln_fse,
                 phi_fse=phi_fse, phi_a=phi_a, perc_a=perc_a, indArr=indArr,
                 nodesComplete=nodesComplete)
ray.shutdown()
np.savez(f'PyDRex{dim}D_{name}_Complete', radani=radani, percani=percani,
         ln_fse=ln_fse, phi_fse=phi_fse, phi_a=phi_a, perc_a=perc_a)

end = perf_counter()
hours = int((end - begin) / 3600)
minutes = int((end - begin - hours * 3600) / 60)
seconds = end - begin - hours * 3600 - minutes * 60
print(f'{hours}:{minutes}:{seconds}')
