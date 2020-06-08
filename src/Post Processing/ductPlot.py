import math
from itertools import starmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
from scipy.interpolate import RegularGridInterpolator

from fileTreat import *

# get macroscopics info from step saved
steps = getMacrSteps() 
macr = getMacrsFromStep(steps[-1]) # get macroscopics from last step
# Get simulation info
info = getSimInfo()

# Get uz mean velocity in xy plan
uz_mean = macr['uz'].mean(axis=2)

NY, NX = info["NY"], info["NX"]
# dx, dy normalized to 0, 1
dx, dy = 1/(NX-1), 1/(NY-1)
# x, y normalized to -1, 1
x, y = np.arange(-1, 1+0.1*dx, 2*dx), np.arange(-1, 1+0.1*dx, 2*dy)

# Interpolating function to get uz velocity in any (-1, 1) point in plan
interpolating_function = RegularGridInterpolator((x, y), uz_mean)

# Delta teta to consider
dteta = math.atan(dy)
# delta R to consider
dR = dy

# List of radius values
radius_values = np.arange(0, R+1e-10, dR)
# List of teta values
teta_values = np.arange(0, 2*np.pi, dteta)

def get_xy_from_r_teta(r, teta):
    """ Get x, y from R and teta values """
    x = r*np.cos(teta)
    y = r*np.sin(teta)
    return x, y

# Figure in matplotlib
fig = plt.figure()
# Polar plot for velocity
uzPlot = plt.subplot(projection="polar")

# List of uz interpolation values for every radius and teta combination 
# defined by "radius values" and "teta values"
uz_radial = np.array([ # ugly but it works
    [interpolating_function(get_xy_from_r_teta(r, teta)) 
        for r in radius_values] 
            for teta in teta_values]).swapaxes(0, 1)

# Density plot for uz
c = uzPlot.pcolormesh(teta_values, radius_values, 
    uz_radial, cmap='OrRd', vmin=np.min(uz_mean), vmax=np.max(uz_mean))

# configure labels and axis limits
uzPlot.axis([teta_values.min(), teta_values.max(), 
             radius_values.min(), radius_values.max()])

l_teta, lb_teta = plt.thetagrids([i for i in np.arange(0, 360, 45)], fmt=None)
l_radius, lb_radius = plt.rgrids([i for i in np.arange(0, 1.01, 0.2)], fmt=None)

# configure colorbar of the plot
fig.colorbar(c, ax=uzPlot)

# Show plots
plt.show()


def get_uz_values_from_r(r):
    """ Get list of uz values with radius R """
    global dR, uz_radial
    return uz_radial[int(r/dR), :]

def get_uz_values_from_teta(teta):
    """ Get list of uz values with angle Teta """
    global dteta, uz_radial
    return uz_radial[:, int(teta/dteta)]

def get_uz_values_from_r_and_teta(r, teta):
    """ Get uz value with angle Teta and radius R """
    global dteta, dR, uz_radial
    return uz_radial[int(r/dR), int(teta/dteta)]

# Example of above functions usage
print("r=0:", get_uz_values_from_r(0))
print("teta=pi/4:", get_uz_values_from_teta(np.pi/4))
print("r=0, teta=pi/4:", get_uz_values_from_r_and_teta(0.2, np.pi/2))