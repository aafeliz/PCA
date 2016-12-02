cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project(PCA)

# link to my header(.h) files
include_directories(include)

# now to create var for all my source(.cpp) files
set(MYSRC src/Matrix.cpp)# src/PCA.cpp)
# FOR FUTURE USE THE FOLLOWING
#file(GLOB MYSRC "src/*.cpp")

set(MATRIX_SRC src/Matrix.cpp)
set(SOURCE_FILES main.cpp ${MATRIX_SRC})


add_executable(PCA ${SOURCE_FILES})