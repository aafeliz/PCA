import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Input Graph
csv = np.genfromtxt ('dataelv.csv', delimiter=",")
#ifirst = csv[:,0]
#isecond = csv[:,1]
#ithird = csv[:,2]

ifirst = csv[0,:]
isecond = csv[1,:]
ithird = csv[2,:]

colors = ["blue"]*50+["red"]*50

a = np.array([-0.682,0.704,-0.198])
b = np.array([0.078,0.338,0.938])

point1  = np.array([0,0,0])
normal1 = np.cross(a,b)

d1 = -np.sum(point1*normal1)# dot product

 #create x,y
xx, yy = np.meshgrid(range(30), range(30))

 #calculate corresponding z
z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx,yy,z1, color='blue', rstride = 5, cstride = 5, shade = 'false')
fig = plt.figure()
ax = Axes3D(fig)

plt3d.scatter(ifirst,isecond,ithird)
#ax.scatter(ifirst,isecond,ithird, c = colors)

xs = [110, 110 - 0.682*50]
ys = [25, 25 + .704*50]
zs = [50, 50 - .198*50]
ax.plot(xs,ys,zs)

xs1 = [110, 110 + .078*50]
ys1 = [25, 25 + .338*50]
zs1 = [50, 50 + .938*50]
ax.plot(xs1,ys1,zs1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#Output Graph
csv = np.genfromtxt ('output.csv', delimiter=",")
ofirst = csv[1,:]
osecond = csv[0,:]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(ofirst,osecond)

ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title("Result Graph")

plt.show()
