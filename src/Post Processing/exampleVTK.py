from dataSave import *
from fileTreat import *
import math

# Get the macroscopics in the folder
macrSteps = getMacrSteps()
info = getSimInfo()

# for all steps saved
for step in macrSteps:
    macr = getMacrsFromStep(step)
    # Save macroscopics to VTK format
    print("Processing step", step)
    saveVTK3D(macr, info['ID'] + "macr" + str(step).zfill(6), points=False)

    # COUETTE/PARALLEL PLATES PROCESSING
    '''
    uz = np.array([np.average(macr['uz'][:, y, info['NZ']//2]) for y in range(0, info['NY'])])
    saveMacrCsv(info['ID'] + "uz" + str(step) + ".csv", uz)
    '''

    # CIRCULAR DUCT PROCESSING
    '''
    R = info['NY']/2
    uzS_N = np.array([np.average(macr['uz'][info['NX']//2, y, :]) 
        for y in range(0, info['NY'])])
    saveMacrCsv(info['ID'] + "uzN-S" + str(step) + ".csv", uzS_N)

    def get_r_times_dr(y, R):
        # dr = 1
        diff = info['NY']/2-R
        # +0.5 due to wet node
        r = abs(y + 0.5 - R) + diff
        return r

    def get_avg_vel(uz_x_half, R):
        uz_integral = 0  # integral(0, r)(u*r*dr)
        for y in range(0, int(np.floor(R))):
            r = get_r_times_dr(y, R)
            uz_integral += r*uz_x_half[y]
        uz_avg = 2/(R*R)*uz_integral  # 2/R^2*integral(0, r)(u*r*dr)
        return uz_avg

    if((info["NY"]/2-R+0.5) == 0.5):  # Correct velocity for wall velocity if node is halfway from wall
        u_correct = (3*uzS_N[0]-uzS_N[1])/2
        uzS_N = [uz - u_correct for uz in uzS_N]
    avg_uz = get_avg_vel(uzS_N, R)
    print("uz_avg", "N %d %.6e" % (info['NY'], avg_uz))
    '''

    # LID DRIVEN PROCESSING
    '''
    # get uy(x=[0, 1], y=NY/2, z=NZ/2) and uy(x=NX/2, y=[0, 1], z=NZ/2)
    uy = macr['uy'][:, info['NY']//2, info['NZ']//2]
    ux = macr['ux'][info['NX']//2, :, info['NZ']//2]
    
    saveMacrCsv(info['ID'] + "uy" + str(step) + ".csv", uy)
    saveMacrCsv(info['ID'] + "ux" + str(step) + ".csv", ux)
    '''

    # SQUARE DUCT PROCESSING
    '''
    # get uz(x=[0, 1], y=[0, 1], z=NZ/2)
    uz = macr['uz'][:,:,info['NZ']//2]
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