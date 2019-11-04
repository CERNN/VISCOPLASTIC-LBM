import os
import glob
import numpy as np

# ALL FILES IN THE FOLDER MUST BE FROM THE SAME SIMULATION
PATH = "./"
#PATH = "./../CUDA/bin/parallelPlatesHWBB/001/"
__macr_names__ = ['ux', 'uy', 'uz', 'rho']

__info__ = dict()

'''
    @brief Get all macroscopics filenames from the folder
    @param macrName (str): macroscopic name ('ux', 'uy', 'uz', 'rho')
    @return list of macroscopics filenames
'''
def getFilenamesMacr(macrName):
    listFiles = glob.glob(PATH+"*"+macrName+"*.bin")
    listFiles.sort()
    return listFiles


'''
    @brief Get all macroscopics steps from the folder
    @return sorted list of steps values
'''
def getMacrSteps():
    listFiles = getFilenamesMacr(__macr_names__[0])
    setMacrSteps = set()
    for i in listFiles:
        macrStep = i.split(__macr_names__[0])[-1] # get everything after the macr name
        macrStep = macrStep[:-4] # take off ".bin"
        macrStep = int(macrStep) # convert to int
        setMacrSteps.add(macrStep) # add macr step to set
    listMacrSteps = list(setMacrSteps)
    listMacrSteps.sort()
    return listMacrSteps



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
    @brief Read the binary file and returns its content as a 3D matrix
    @param macrFilename (str): filename of the macroscopic
    @return macroscopic array (np.array([x, y, z]))
'''
def readFileMacr3D(macrFilename):
    info = getSimInfo()
    if info['Prc'] == 'double':
        prc = 'd'
    elif info['Prc'] == 'float':
        prc = 'f'
    with open(macrFilename, "r") as f:
        vec = np.fromfile(f, prc)
        vec3D = np.reshape(vec, (info['NZ'], info['NY'], info['NX']), 'C')
        return np.swapaxes(vec3D, 0, 2)


'''
    @brief Get all macroscopics in the folder from the step specified
    @param step: step of the macroscopics
    @return Dictionary with list of macroscopics  (dict(np.array()))
'''
def getMacrsFromStep(step):
    macr = dict()
    listFilename = list()
    for macrName in __macr_names__:
        listFilename.append(getFilenamesMacr(macrName))
    # flatten list
    listFilenameFlat = [j for i in range(0, len(listFilename)) \
        for j in listFilename[i]]
    # get all filenames of the step
    listFilenameStep = [i for i in listFilenameFlat if (("%05d" % step)+".bin") in i]
    # if there is no macroscopic from that step
    if len(listFilenameStep) == 0:
        return None

    for filename in listFilenameStep:
        for macrName in __macr_names__:
            if macrName in filename:
                macr[macrName] = readFileMacr3D(filename)

    return macr

'''
    @brief Get array for all macroscopics in folder. For compatibility all 
            macrs must have the same number of files and same steps for each
    @return Dictionary with list of macroscopics  (dict(list(np.array())))
'''
def getAllMacrs():
    macr = dict()

    filenames = dict()
    for macrName in __macr_names__:
        filenames[macrName] = getFilenamesMacr(macrName)

    for i in range(0, min(len(filenames[i]) for i in filenames)):
        # getting simulation step
        macrStep = filenames[__macr_names__[0]][i].split(__macr_names__[0])[-1]
        macrStep = macrStep[:-4] # take off ".bin"
        macrStep = int(macrStep)
        
        # save macroscopics from simulation step
        macr[macrStep] = dict()
        for macrName in filenames:
            macr[macrStep][macrName] = readFileMacr3D(filenames[macrName][i])
    
    return macr

