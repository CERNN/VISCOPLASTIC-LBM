import math

SCALE = 1
NX, NY, NZ = 128, 128, 128

U_MAX = 1
#PATH_DATA = "./../CUDA/simulations/D3Q19/dump/"
PATH_DATA = "./"
ID_SIM = "020"
SEP = ','
#STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#STEPS = [x for x in range(10, 101, 10)]
STEPS = [100]

# obs.: the "str(0) * math.floor(math.log((1e6 - 1) / steps, 10))" part comes from the format that the file is saved
PATH_RHO_READ = [PATH_DATA + ID_SIM + '_rho' + str(0) * math.floor(math.log((1e6 - 1)/ steps, 10)) + str(steps)+ '.bin' for steps in STEPS]
PATH_RHO_WRITE = [PATH_DATA + ID_SIM + '_rho' + str(0) * math.floor(math.log((1e6 - 1)/ steps, 10))+ str(steps) + '.csv' for steps in STEPS]

PATH_UX_READ = [PATH_DATA + ID_SIM + '_ux' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10))+ str(steps) + '.bin' for steps in STEPS]
PATH_UX_WRITE = [PATH_DATA + ID_SIM + '_ux' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10))+ str(steps) + '.csv' for steps in STEPS]

PATH_UY_READ = [PATH_DATA + ID_SIM + '_uy' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10)) + str(steps) + '.bin' for steps in STEPS]
PATH_UY_WRITE = [PATH_DATA + ID_SIM + '_uy' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10)) + str(steps) + '.csv' for steps in STEPS]


PATH_UZ_READ = [PATH_DATA + ID_SIM + '_uz' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10)) + str(steps) + '.bin' for steps in STEPS]
PATH_UZ_WRITE = [PATH_DATA + ID_SIM + '_uz' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10)) + str(steps) + '.csv' for steps in STEPS]

PATH_WRITE_VTK = [PATH_DATA + ID_SIM + '_macr' + str(0) * math.floor(math.log((1e6 - 1) / steps, 10)) + str(steps) for steps in STEPS]