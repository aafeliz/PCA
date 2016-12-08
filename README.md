# PCA
# A general Principal Component Analysis class
PCA is popular method for dealing with the "curse of dimensionality".

It is a linear transform that provides the best projection that can represent a class; By using least square method onto its different features and associated data.

# GUI
Features our teaching method, that will show of what happens to the data under various steps of PCA. In addition the different collection of statistics and information gained by PCA.

# Data
To ensure valid results, we will choose a popular data set and compare our results with the results of PCA on matlab or python sklearn.

# Build Instructions
To build an executable in a UNIX environment, open Terminal navigate to PCA directory and type:
    $ g++ -std=c++11 -o PCA main.cpp src/Matrix.cpp src/PCA.cpp
