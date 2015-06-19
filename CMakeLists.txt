cmake_minimum_required(VERSION 2.8)
project( fluidSim )

find_package( OpenCV REQUIRED )

set(EXECUTABLE_OUTPUT_PATH ../bin)
set (CMAKE_CXX_FLAGS "-Wall -Werror -pedantic -g -std=c++11") 

set ( Project_SOURCES
    ../src/CPU.cpp
    ../src/main.cpp
    ../src/FluidSimulation.cpp
    )
    
add_executable( fluidSim ${Project_SOURCES} )

target_link_libraries( fluidSim ${OpenCV_LIBS})

