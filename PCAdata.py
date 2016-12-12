import numpy as np

data = np.zeros([50,100])

for i in range(100):
    mean = np.random.uniform(150)
    sd = np.random.uniform(10)
    col = np.random.normal(mean,sd,50)
    data[:,i] = col

print "data1: "
print data

data2 = np.zeros([50,100])

for i in range(100):
    mean = np.random.uniform(150)
    sd = np.random.uniform(10)
    col = np.random.normal(mean,sd,50)
    data2[:,i] = col

print "data2: "
print data2

data3 = np.concatenate((data,data2))

print "data3: "
print data3

np.savetxt("data.csv",data3, delimiter=",")

data4 = data3.transpose()

np.savetxt("data2.csv",data4, delimiter=",")
