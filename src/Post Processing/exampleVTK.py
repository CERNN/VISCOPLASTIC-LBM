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

    # Save uy(x=[0, 1], y=NY/2, z=0.5) to csv
    uyX = macr[step]['uy']
    uyZ = macr[step]['uy']
    # uz[y] <= average(uz[:, y, z=NZ/2])
    uyX = [(uy[x, info['NY']//2, info['NZ']//2]) for x in range(0, info['NX'])]
    uyZ = [(uy[info['NX']//2, info['NY']//2, z]) for z in range(0, info['NZ'])]

    saveMacrLineCsv(info['ID'] + "uyX" + str(step) + ".csv", uyX)
    saveMacrLineCsv(info['ID'] + "uyZ" + str(step) + ".csv", uyZ)
    
    # analytical solution for square duct
    uyXAnalytical = np.zeros(uyX.shape)
    uyZAnalytical = np.zeros(uyZ.shape)
    for i in range(0, len(uyXAnalytical)):
        x = (i + 0.5) / (info['NX']) # node is delocated 0.5
        z = 0.5 + 0.5/info['NZ'] # node is delocated from center
        uyXAnalytical[x]