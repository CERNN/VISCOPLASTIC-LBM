from pyevtk.hl import gridToVTK
from fileTreat import *

'''
    TODO: add support to POINT DATA
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
    @brief Saves macroscopics in a line in a csv file
    @param filenameWrite (str): filename to write to
    @param macrLine (np.array()): array with line of macroscopics to save
    @param normalize (bool): normalize distance or not
'''
def saveMacrLineCsv(filenameWrite, macrLine, normalize=False):
    with open(PATH+filenameWrite, 'w') as f:
        if(not normalize):
            np.savetxt(f, [(i, macrLine[i]) for i in range(0, len(macrLine))], \
                fmt=['%d', '%.6e'], delimiter=',')
        else:
            np.savetxt(f, [(i/len(macrLine), macrLine[i]) for i in range(0, len(macrLine))], \
                fmt=['%d', '%.6e'], delimiter=',')