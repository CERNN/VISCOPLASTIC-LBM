import os
import glob
import numpy as np

# ALL FILES IN THE FOLDER MUST BE FROM THE SAME SIMULATION
PATH = "./../CUDA/bin/parallelPlatesHWBB/001/"
#PATH = "./"

__info__ = dict()

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
    if len(__info__) == 0:
        filename = glob.glob(PATH+"*info*.txt")[0]
        with open(filename, "r") as f:
            lines = f.readlines()
            linesTrim = [l.strip() for l in lines]
            
            try:
                __info__['ID'] = [str(txt.split(" ")[-1]) for txt in linesTrim \
                    if 'Simulation ID' in txt][0]
            except:
                print("Not able to get ID from info file")
            
            try:
                __info__['Prc'] = [txt.split(" ")[-1] for txt in linesTrim if 'Precision' in txt][0]
            except:
                print("Not able to get Precision from info file")
            
            try:
                __info__['NX'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NX' in txt][0]
            except:
                print("Not able to get NX from info file")
            
            try:
                __info__['NY'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NY' in txt][0]
            except:
                print("Not able to get NY from info file")
            
            try:
                __info__['NZ'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'NZ' in txt][0]
            except:
                print("Not able to get NZ from info file")
            
            try:
                __info__['Tau'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'Tau' in txt][0]
            except:
                print("Not able to get Tau from info file")
            
            try:
                __info__['Umax'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'Umax' in txt][0]
            except:
                print("Not able to get Umax from info file")
            
            try:
                __info__['Nsteps'] = [int(txt.split(" ")[-1]) for txt in linesTrim if 'Nsteps' in txt][0]
            except:
                print("Not able to get Nsteps from info file")
            
            try:
                __info__['FX'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FX' in txt][0]
            except:
                print("Not able to get FX from info file")
            
            try:
                __info__['FY'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FY' in txt][0]
            except:
                print("Not able to get FY from info file")
            
            try:
                __info__['FZ'] = [float(txt.split(" ")[-1]) for txt in linesTrim if 'FZ' in txt][0]
            except:
                print("Not able to get FZ from info file")

    return __info__


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
    @brief Get array for all macroscopics in folder for compatibility all 
            macrs must have the same number of files and same steps for each
    @return Dictionary with list of macroscopics  (dict(list(np.array())))
'''
def getAllMacr3D():
    info = getSimInfo()
    if info['Prc'] == 'double':
        prc = 'd'
    elif info['Prc'] == 'float':
        prc = 'f'
    macr = dict()

    fileNames = dict()
    fileNames['ux'] = getFilenamesMacr('ux')
    fileNames['uy'] = getFilenamesMacr('uy')
    fileNames['uz'] = getFilenamesMacr('uz')
    fileNames['rho'] = getFilenamesMacr('rho')

    for i in range(0, min(len(fileNames[i]) for i in fileNames)):
        # getting simulation step
        macrStep = fileNames['ux'][i].split("ux")[-1]
        macrStep = macrStep[:-4] # take off ".bin"
        macrStep = int(macrStep)
        
        # save macroscopics from simulation step
        macr[macrStep] = dict()
        for macrName in fileNames:
            macr[macrStep][macrName] = getMacr3D(fileNames[macrName][i], prc)
    
    return macr

    