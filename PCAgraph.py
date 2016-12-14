import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Input Graph
csv = np.genfromtxt ('data.csv', delimiter=",")
ifirst = csv[:,0]
isecond = csv[:,1]
ithird = csv[:,2]
ifourth = csv[:,3]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(ifourth,isecond,ithird)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#Output Graph
csv = np.genfromtxt ('matrix.csv', delimiter=",")
ofirst = csv[:,0]
osecond = csv[:,1]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(ifourth,osecond)

ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title("Result Graph")

plt.show()
