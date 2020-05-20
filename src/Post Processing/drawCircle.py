import matplotlib.pyplot as plt
import numpy as np

# R = NY/2.0-1;
# xCenter = (NX/2.0);
# yCenter = (NY/2.0);
N = 17 #grid size
R = N/2-1 #radius
C = (N/2, N/2) # center

fig, ax = plt.subplots()

ax.set(xlim=(0, N), ylim = (0, N))

a_circle = plt.Circle(C, R, fill=False)
ax.add_artist(a_circle)
ax.set_xticks(np.arange(0, N, 1))
ax.set_yticks(np.arange(0, N, 1))
ax.grid()
fig.show()
for x in np.arange(0.5, N, 1):
    for y in np.arange(0.5, N, 1):
        dist = ((x-C[0])**2+(y-C[0])**2)**0.5
        if(dist > R):
            ax.add_artist(plt.Circle((x, y), 0.1, color='r'))
        elif(dist > (R-(2**0.5))):
            print(x, y, R-dist)
            ax.add_artist(plt.Circle((x, y), 0.1, color='y'))
        elif(dist > (R-2*(2**0.5))):
            ax.add_artist(plt.Circle((x, y), 0.1, color='g'))
        else:
            ax.add_artist(plt.Circle((x, y), 0.1, color='b'))
input()
