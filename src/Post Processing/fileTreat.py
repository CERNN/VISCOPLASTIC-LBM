import os
import glob
import numpy as np

PATH = "./../CUDA/bin/parallelPlates/003/"

'''
    @brief Get all macroscopics filenames from this folder
    @param macrName (str): macroscopic name ('ux', 'uy', 'uz', 'rho')
    @return list of macroscopics filenames
'''
def getFilenamesMacr(macrName):
    listFiles = glob.glob(PATH+"*"+macrName+"*.bin")
    listFiles.sort()
    return listFiles


'''
    @brief Get simulation info in dictionary format
    @return dictionary with simulation info
'''
def getSimInfo():
    filename = glob.glob(PATH+"*info*.txt")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        linesTrim = [l.strip() for l in lines]
        info = dict()
        info['ID'] = [str(txt.split(" ")[-1]) for txt in linesTrim \
            if 'Simulation ID' in txt][0]
        info['NX'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NX' in txt][0]
        info['NY'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NY' in txt][0]
        info['NZ'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NZ' in txt][0]
        info['Tau'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'Tau' in txt][0]
        info['Umax'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'Umax' in txt][0]
        info['FX'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FX' in txt][0]
        info['FY'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FY' in txt][0]
        info['FZ'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FZ' in txt][0]
        info['Nsteps'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'Nsteps' in txt][0]
        info['Prc'] = [txt.split(" ")[-1] for txt in linesTrim if 'Precision' in txt][0]
    return info


'''
    @brief Get the macroscopic array in 3D
    @param macrFilename (str): filename of the macroscopic
    @param prcNp (str): precision of the macroscopic ('d': double, 'f': float)
    @return macroscopic array (np.array([x, y, z]))
'''
def getMacr3D(macrFilename, prcNp):
    info = getSimInfo()
    with open(macrFilename, "r") as f:
        vec = np.fromfile(f, prcNp)
        vec3D = np.reshape(vec, (info['NZ'], info['NY'], info['NX']), 'C')
        return np.swapaxes(vec3D, 0, 2)


'''
    @brief Get array for each macroscopic
           (use when there is only one file for each macroscopic in folder)
    @return Dictionary with macroscopics (dict(np.array()))
'''
def getEachMacr3D():
    info = getSimInfo()
    if info['Prc'] == 'double':
        prc = 'd'
    elif info['Prc'] == 'float':
        prc = 'f'
    macr = dict()
    macr['ux'] = getMacr3D((getFilenamesMacr('ux'))[-1], prc)
    macr['uy'] = getMacr3D(getFilenamesMacr('uy')[-1], prc)
    macr['uz'] = getMacr3D(getFilenamesMacr('uz')[-1], prc)
    macr['rho'] = getMacr3D(getFilenamesMacr('rho')[-1], prc)
    return macr


'''
    @brief Get array for all macroscopics in folder
           (use when there are many files in folder for each macroscopic)
    @return Dictionary with list of macroscopics  (dict(list(np.array())))
'''
def getAllMacr3D():
    info = getSimInfo()
    if info['Prc'] == 'double':
        prc = 'd'
    elif info['Prc'] == 'float':
        prc = 'f'
    macr = dict()

    macr['ux'] = [getMacr3D(f, prc) for f in getFilenamesMacr('ux') if f != None]
    macr['uy'] = [getMacr3D(f, prc) for f in getFilenamesMacr('uy') if f != None]
    macr['uz'] = [getMacr3D(f, prc) for f in getFilenamesMacr('uz') if f != None]
    macr['rho'] = [getMacr3D(f, prc) for f in getFilenamesMacr('rho') if f != None]
    return macr

    