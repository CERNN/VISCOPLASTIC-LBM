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
# dx, dy normalized to 0, 2
dx, dy = 2/(NX-1), 2/(NY-1)
# x, y normalized to -1, 1
x, y = np.arange(-1, 1+0.1*dx, dx), np.arange(-1, 1+0.1*dy, dy)

# Interpolating function to get uz velocity in any (-1, 1) point in plan
interpolating_function = RegularGridInterpolator((x, y), uz_mean)

# delta R to consider
dR = dy
# Delta teta to consider (number of dR possible values times pi)
dteta = 2*np.pi/(1/dR) / np.pi

# List of radius values (normalized from 0 to 1)
radius_values = np.arange(0, 1+1e-10, dR)
# List of teta values
teta_values = np.arange(0, 2*np.pi, dteta)
print(radius_values.shape, teta_values.shape)

def get_xy_from_r_teta(r, teta):
    """ Get x, y from R and teta values """
    x = r*np.cos(teta)
    y = r*np.sin(teta)
    return x, y

# Figure in matplotlib
fig1 = plt.figure(1)
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

l_teta, lb_teta = plt.thetagrids([i for i in np.arange(0, 360, 45)], fmt=None)
l_radius, lb_radius = plt.rgrids([i for i in np.arange(0, 1.01, 0.2)], fmt=None)

# configure colorbar of the plot
fig1.colorbar(c, ax=uzPlot)

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

# List of uz values for different R
uz_in_r = np.array([get_uz_values_from_r(0.5), # R=0.5
            get_uz_values_from_r(0.1), # R=0.1
            get_uz_values_from_r(0.9)]) # R=0.9

# List of uz values for different teta
uz_in_teta = np.array([
    get_uz_values_from_teta(np.pi/4), # teta = 45 degrees
    get_uz_values_from_teta(np.pi/12), # teta = 15 degrees
    get_uz_values_from_teta(0) # teta = 0 degree
])

uz_in_r_and_teta = get_uz_values_from_r_and_teta(0.2, np.pi/2) #r = 0.2, pi=0.2

# Plot uz in r
fig2 = plt.figure(2)
uz_r_plot = plt.subplot()
# Plot
for uz_r in uz_in_r:
    uz_r_plot.plot(teta_values, uz_r)

# configure labels and axis limits
uz_r_plot.set(xlabel='teta', ylabel='uz', ylim=(0, np.max(uz_in_r)*1.1),
              xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ["0", "$\\frac{\pi}{2}$", "$\pi$", "$\\frac{3\pi}{2}$", "$2\pi$"])

# Plot uz in teta
fig3 = plt.figure(3)
uz_teta_plot = plt.subplot()
# Plot
for uz_teta in uz_in_teta:
    uz_teta_plot.plot(radius_values, uz_teta, linewidth=2)

# configure labels and axis limits
uz_teta_plot.set(xlabel='R', ylabel='uz', xlim=(0,1), 
                ylim=(0, np.max(uz_in_teta)*1.1),
                xticks=[0, 0.5, 1.0])

plt.show()
