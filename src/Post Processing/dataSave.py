from pyevtk.hl import gridToVTK
from fileTreat import *


def saveVTK3D(macrsDict, filenameWrite, points=True, normVal=1):
    """ Saves variables values to VTK format

    Parameters
    ----------
    macrsDict : dict()
        Dict with variable values and name as key
    filenameWrite : str
        Filename to write to (NO EXTENSION)
    points : bool, optional
        If True, save as point centered data, if False,
        save as cell centered data, by default True
    normVal : int, optional
        Value to normalize distance (if zero, the distance is
        normalized by NX), by default 0
    """

    info = getSimInfo()

    if(normVal == 0):
        normVal = info['NX']
        if(points == True):
            normVal -= 1

    dx, dy, dz = 1.0 / normVal, 1.0 / normVal, 1.0 / normVal
    if info['Prc'] == 'double':
        prc = 'float64'
    elif info['Prc'] == 'float':
        prc = 'float32'

    if(points == False):
        # grid
        x = np.arange(0, info['NX'] / normVal + 0.1 * dx, dx, dtype=prc)
        y = np.arange(0, info['NY'] / normVal + 0.1 * dy, dy, dtype=prc)
        z = np.arange(0, info['NZ_TOTAL'] / normVal + 0.1 * dz, dz, dtype=prc)
        gridToVTK(PATH + filenameWrite, x, y, z, cellData=macrsDict)
    else:
        # grid
        x = np.arange(0, (info['NX'] - 1) / normVal + 0.1 * dx, dx, dtype=prc)
        y = np.arange(0, (info['NY'] - 1) / normVal + 0.1 * dy, dy, dtype=prc)
        z = np.arange(0, (info['NZ_TOTAL'] - 1) / normVal + 0.1 * dz, dz, dtype=prc)
        gridToVTK(PATH + filenameWrite, x, y, z, pointData=macrsDict)


def saveMacrCsv(filenameWrite, macr, normalizeDist=False):
    """ Saves macroscopics in a csv file

    Parameters
    ----------
    filenameWrite : str
        Filename to write to
    macr : np.array()
        Array with macroscopics to save (1D or 2D)
    normalizeDist : bool, optional
        Normalize distance or not for 1D, by default False
    """

    with open(PATH + filenameWrite, 'w') as f:
        if(len(macr.shape) == 1):  # 1D
            # csv is: position, value
            if(not normalizeDist):
                array = [[i, macr[i]] for i in range(0, len(macr))]
                np.savetxt(f, array, fmt=['%d', '%.6e'], delimiter=',')
            else:
                array = [(i / len(macr), macr[i]) for i in range(0, len(macr))]
                np.savetxt(f, array, fmt=['%.6e', '%.6e'], delimiter=',')
        elif(len(macr.shape) == 2):  # 2D
            np.savetxt(f, macr, delimiter=',')
        else:
            print("Input array for \"saveMacrCsv\" is not 2D or 1D")
