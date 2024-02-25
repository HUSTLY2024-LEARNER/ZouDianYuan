# cmake version
cmake_minimum_required(VERSION 3.15)

# project name
project(HorizonRobot)

# package requirement
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# set compiler version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# optimize option
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE -Ofast)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")

# include dir add
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./includes)

# link dir add
# link_directories()

# output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# add source
aux_source_directory(./src DIR_SRCS)

# add executable
add_executable(robot ${DIR_SRCS})

# link add

target_link_libraries(robot Threads::Threads)
target_link_libraries(robot  ${OpenCV_LIBS} )