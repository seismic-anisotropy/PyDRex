#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvalsh
from scipy.integrate import solve_ivp
from scipy.interpolate import NearestNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import vtk
from time import perf_counter
import warnings

from DRexParam import gridMax, gridMin, size


def interpVel(pointCoords, dictGlobals=None):
    pointVel = np.array(
        [iVelX(*pointCoords),
         (iVelY(*pointCoords) if len(pointCoords) == 3 else np.nan),
         -iVelZ(*pointCoords)])
    return pointVel[~np.isnan(pointVel)]


def interpVelGrad(pointCoords, dictGlobals=None, ivpIter=False):
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
    assert abs(np.sum(np.diag(L))) < 1e-15, [L, pointCoords]
    if ivpIter:
        return L
    e = (L + np.transpose(L)) / 2
    epsnot = np.amax(np.absolute(eigvalsh(e)))
    return L, epsnot


def pathline(currPoint, dictGlobals, ivpMethod, first_step, max_step, t_eval,
             atol, rtol):
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
                        method=ivpMethod, first_step=first_step,
                        max_step=max_step, t_eval=t_eval, events=[maxStrain],
                        args=(dictGlobals,), dense_output=True, jac=ivpJac,
                        atol=atol, rtol=rtol)
    return sol.t, sol.sol


def isInside(point, dictGlobals=None):
    inside = True
    for coord, minBound, maxBound in zip(point, gridMin, gridMax):
        if coord < minBound or coord > maxBound:
            inside = False
            break
    return inside


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''
A simple plotting script using Matplotlib to generate filled contour plots.
Requires the input file DRexParam.py in the same directory.''')
parser.add_argument('input', help='input file (expects a VTK unstructured grid)')
args = parser.parse_args()

vtkReader = vtk.vtkXMLUnstructuredGridReader()
vtkReader.SetFileName(args.input)
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
    tri = Delaunay(coords[:, :-1])
    iVelX = CloughTocher2DInterpolator(tri, vel[:, 0])
    iVelZ = CloughTocher2DInterpolator(tri, vel[:, 1])
    iLxx = CloughTocher2DInterpolator(tri, velGrad[:, 0])
    iLzx = CloughTocher2DInterpolator(tri, velGrad[:, 1])
    iLxz = CloughTocher2DInterpolator(tri, velGrad[:, 3])
    iLzz = CloughTocher2DInterpolator(tri, velGrad[:, 4])
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
del vtkOut

alt = np.zeros((3, 3, 3))  # \epsilon_{ijk}
for ii in range(3):
    alt[ii % 3, (ii + 1) % 3, (ii + 2) % 3] = 1
    alt[ii % 3, (ii + 2) % 3, (ii + 1) % 3] = -1
# ijkl -> Tensor of indices to form Cijkl from Sij
ijkl = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype='int')
# l1, l2 -> Tensors of indices to form Sij from Cijkl
l1 = np.array([0, 1, 2, 1, 2, 0], dtype='int')
l2 = np.array([0, 1, 2, 2, 0, 1], dtype='int')
# Direction cosine matrix with uniformly distributed rotations.
acs0 = R.random(size, random_state=1).as_matrix()


method = ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']
plotTime = -1e4 * 365.25 * 8.64e4 * np.arange(10001)
if dim == 3:
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    point = [2.4e6, 7e5, 2e4]
else:
    fig, ax = plt.subplots()
    point = [1.1e6, 1e5]
for name in method:
    begin = perf_counter()
    pathTime, pathDense = pathline(
        point, None, name, first_step=1e11, max_step=np.inf, t_eval=None,
        atol=1e-8, rtol=1e-5)
    end = perf_counter()
    if dim == 3:
        ax.plot3D(*pathDense(plotTime[plotTime > pathTime[-1]]),
                  label=f'{name}: {end - begin:.3f}s')
    else:
        ax.plot(*pathDense(plotTime[plotTime > pathTime[-1]]),
                label=f'{name}: {end - begin:.3f}s')
    begin = perf_counter()
    pathTime, pathDense = pathline(
        point, None, name, first_step=None, max_step=5e11, t_eval=None,
        atol=1e-9, rtol=1e-6)
    end = perf_counter()
    if dim == 3:
        ax.plot3D(*pathDense(plotTime[plotTime > pathTime[-1]]),
                  label=f'{name}: {end - begin:.3f}s', linestyle='dashed')
    else:
        ax.plot(*pathDense(plotTime[plotTime > pathTime[-1]]),
                label=f'{name}: {end - begin:.3f}s', linestyle='dashed')
ax.invert_yaxis()
ax.set_aspect('auto') if dim == 3 else ax.set_aspect('equal')
ax.legend(ncol=6)
plt.show()
