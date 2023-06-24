import numpy as np

gridres = np.array([4e3, 4e3])
gridmin = np.array([0, 0])
gridmax = np.array([1.2e6, 4e5]) + gridres
gridnodes = ((gridmax - gridmin) / gridres + 1).astype(int)
gridcoords = [np.linspace(x, y, z) for x, y, z in zip(gridmin, gridmax, gridnodes)]
