cmake_minimum_required(VERSION 3.8)
project(segmentation)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCE_FILES Mission2.b1.cpp)

add_executable(main Mission2.b1.cpp)
target_link_libraries(main  ${OpenCV_LIBS} )