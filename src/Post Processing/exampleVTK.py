from dataSave import *
import math

# Get the macroscopics in the folder
macr = getAllMacr3D()
info = getSimInfo()

# for all steps saved
for step in macr:
    # Save macroscopics to VTK format
    saveVTK3D(macr[step], info['ID'] + "macr" + str(step), points=False)


    # COUETTE/PARALLEL PLATES PROCESSING
    uz = np.array([np.average(macr[step]['uz'][:, y, info['NZ']//2]) for y in range(0, info['NY'])])
    saveMacrCsv(info['ID'] + "uz" + str(step) + ".csv", uz)


    '''
    # LID DRIVEN PROCESSING
    
    # get uy(x=[0, 1], y=NY/2, z=NZ/2) and uy(x=NX/2, y=[0, 1], z=NZ/2)
    uy = macr[step]['uy'][:, info['NY']//2, info['NZ']//2]
    ux = macr[step]['ux'][info['NX']//2, :, info['NZ']//2]
    
    saveMacrCsv(info['ID'] + "uy" + str(step) + ".csv", uy)
    saveMacrCsv(info['ID'] + "ux" + str(step) + ".csv", ux)
    '''

    '''
    # SQUARE DUCT PROCESSING
    
    # get uz(x=[0, 1], y=[0, 1], z=NZ/2)
    uz = macr[step]['uz'][:,:,info['NZ']//2]
    # mean corrected for free slip in y=0 and x=1
    uzMean = np.average(uz)
    uz /= uzMean # normalize

    # analytical solution for square duct
    dx = 1/(info['NX']-1) # for free slip
    x = np.arange(dx/2, 0.5+0.1*dx, dx) # x
    y = np.arange(dx/2, 0.5+0.1*dx, dx) # y
    uzAnalytical = np.zeros((len(x), len(y)))

    for j in range(0, len(x)):
        uzAnalytical[j,:] = 0.5*x[j]*(1-x[j])
        for k in range(0, len(y)):
            for i in range(1, 100):
                bn = (2*i-1)*math.pi/1
                uzAnalytical[j, k] -= (4/(math.pi**3))* \
                    (1/((2*i-1)**3)*math.sin(bn*x[j]))* \
                    (math.sinh(bn*y[k])+math.sinh(bn*(1-y[k])))/math.sinh(bn)
    mean = 0
    for k in range(1, 201):
        bn = (2*k-1)*math.pi
        mean = mean + np.tanh(bn/2)/((2*k-1)**5)
    meanAnalytical = (1/12)*(1-192*mean/(math.pi**5))
    uzAnalytical /= meanAnalytical
    maxAnalytical = np.max(uzAnalytical)

    maxNumerical = np.max(uz)
    print("Max. Numerical:", maxNumerical)
    print("Max. Analytical:", maxAnalytical)
    print("Error:", str((maxNumerical-maxAnalytical)/maxAnalytical*100) + "%")
    saveMacrCsv(info['ID'] + "uzAnalytical" + str(step) + ".csv", uzAnalytical)
    '''