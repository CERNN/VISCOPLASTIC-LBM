from fileTreat import *
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker

# get macroscopics info from step saved
steps = getMacrSteps()# get macroscopics from last step
macr = getMacrsFromStep(steps[-1])

info = getSimInfo()


''' VELOCITY PLOT '''
# get average uz in x axis, for z=0.5, y in [0, 1]
avgUz = [np.average(macr['uz'][:, y, info['NZ']//2]) for y in range(0, info['NY'])]
# normalize uz by Umax
avgUz = np.divide(avgUz, info['Umax'])
# plot uz(y)
fig, uzPlot = plt.subplots()
# configure y values, normalized to (0, 1)
y = np.arange(0, 1+1e-10, 1/(info['NY']-1))

uzPlot.plot(y, avgUz)
# configure labels and axis limits
uzPlot.set(xlabel='y', ylabel='uz',xlim=(0,1), ylim=(min(avgUz)*1.1, max(avgUz)*1.1))
# configure ticks (print value) for x axis
plt.xticks([0, 0.5, 1.0])

''' DENSITY PLOT (Heatmap) '''
# get rho for plan (x=0.5)
planRho = macr['rho'][info['NX']//2,:,:]
# plot rho(x, y)
fig, rhoPlot = plt.subplots()
# configure x and y values, normalized to (0, 1)
x = np.arange(0, 1+1e-10, 1/(info['NZ']-1)) # x plot is the z simulation axis 
y = np.arange(0, 1+1e-10, 1/(info['NY']-1))

c = rhoPlot.pcolormesh(x, y, planRho, cmap='RdBu', vmin=np.min(planRho), vmax=np.max(planRho))
# configure labels and axis limits
rhoPlot.set_title("Rho")
rhoPlot.set(xlabel='z', ylabel='y')
rhoPlot.axis([x.min(), x.max(), y.min(), y.max()])
# configure colorbar of the plot
fig.colorbar(c, ax=rhoPlot)
# show plot

''' SHOW PLOTS '''
plt.show()