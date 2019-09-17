from dataSave import *

# Get the macroscopics in the folder
macr = getAllMacr3D()
info = getSimInfo()

# for all steps saved
for step in macr:
    # Add velocity norm as a macroscopic
    macr[step]['nmVel'] = np.ndarray(dtype=float, shape=(info['NX'], info['NY'], info['NZ']))
    for x in range(0, info['NX']):
        for y in range(0, info['NY']):
            for z in range(0, info['NZ']):
                macr[step]['nmVel'][x, y, z] = np.linalg.norm([macr[step]['ux'][x, y, z], \
                            macr[step]['uy'][x, y, z], macr[step]['uz'][x, y, z]])
    
    # Save macroscopics to VTK format
    saveVTK3D(macr[step], info['ID'] + "macr" + str(step), points=True)

    # Save average uz(x=[0, 1], y, z=NZ/2) to csv
    uz = macr[step]['uz']
    # uz[y] <= average(uz[:, y, z=NZ/2])
    uz = [np.average(uz[:, y, info['NZ']//2]) for y in range(0, info['NY'])] 

    saveMacrLineCsv(info['ID'] + "uz" + str(step) + ".csv", uz)
