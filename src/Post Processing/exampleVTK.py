from dataSave import *

# Get the macroscopics in the folder
macr = getEachMacr3D()
info = getSimInfo()

# Add velocity norm as a macroscopic
macr['nmVel'] = np.ndarray(dtype=float, shape=(info['NX'], info['NY'], info['NZ']))
for x in range(0, info['NX']):
    for y in range(0, info['NY']):
        for z in range(0, info['NZ']):
            macr['nmVel'][x, y, z] = np.linalg.norm([macr['ux'][x, y, z], \
                        macr['uy'][x, y, z], macr['uz'][x, y, z]])

# Save macroscopics to VTK format
saveVTK3D(macr, info['ID'] + "macr")

# Save average uz(x=[0, 1], y, z=NZ/2) to csv
uz = macr['uz']
# uz[y] <= average(uz[:, y, z=NZ/2])
uz = [np.average(uz[:, y, info['NZ']//2]) for y in range(0, info['NY'])] 

saveMacrLineCsv(info['ID'] + "uz.csv", uz)
