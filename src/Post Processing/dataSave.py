from pyevtk.hl import gridToVTK
from fileTreat import *

'''
    @brief Saves variables values to VTK format
    @param macrsDict (dict()): dict with variable values and name as key 
    @param filenameWrite (str): filename to write to (NO EXTENSION)
    @param points (bool): if True, save as point centered data,
                          if False, save as cell centered data
    @param normVal (float): value to normalize distance. 
                            If zero, the distance is normalized by NX
'''
def saveVTK3D(macrsDict, filenameWrite, points=True, normVal=0):
    info = getSimInfo()
    
    if(normVal == 0):
        normVal = info['NX']
        if(points == True):
            normVal -= 1

    dx, dy, dz = 1.0/normVal, 1.0/normVal, 1.0/normVal
    if info['Prc'] == 'double':
        prc = 'float64'
    elif info['Prc'] == 'float':
        prc = 'float32'
    
    if(points == False):
        # grid
        x = np.arange(0, info['NX']/normVal+0.1*dx, dx, dtype=prc)
        y = np.arange(0, info['NY']/normVal+0.1*dy, dy, dtype=prc)
        z = np.arange(0, info['NZ']/normVal+0.1*dz, dz, dtype=prc)
        gridToVTK(PATH+filenameWrite, x, y, z, cellData=macrsDict)
    else:
        # grid
        x = np.arange(0, (info['NX']-1)/normVal+0.1*dx, dx, dtype=prc)
        y = np.arange(0, (info['NY']-1)/normVal+0.1*dy, dy, dtype=prc)
        z = np.arange(0, (info['NZ']-1)/normVal+0.1*dz, dz, dtype=prc)
        gridToVTK(PATH+filenameWrite, x, y, z, pointData=macrsDict)


'''
    @brief Saves macroscopics in a csv file
    @param filenameWrite (str): filename to write to
    @param macr (np.array()): array with macroscopics to save (1D or 2D)
    @param normalizeDist (bool): normalize distance or not for 1D
'''
def saveMacrCsv(filenameWrite, macr, normalizeDist=False):
    with open(PATH+filenameWrite, 'w') as f:
        if(len(macr.shape) == 1): # 1D
            # csv is: position, value
            if(not normalizeDist):
                np.savetxt(f, [(i, macr[i]) for i in range(0, len(macr))], \
                    fmt=['%d', '%.6e'], delimiter=',')
            else:
                np.savetxt(f, [(i/len(macr), macr[i]) for i in range(0, len(macr))], \
                    fmt=['%d', '%.6e'], delimiter=',')
        elif(len(macr.shape) == 2): # 2D
            np.savetxt(f, macr, delimiter=',')
        else:
            print("Input array for \"saveMacrCsv\" is not 2D or 1D")
