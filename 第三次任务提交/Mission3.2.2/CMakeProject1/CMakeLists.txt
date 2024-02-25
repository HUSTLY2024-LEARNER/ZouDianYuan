find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(/usr/local/include/opencv4)
set(SOURCE_FILES Mission2.b1.cpp)



add_executable (CMakeProject1 "CMakeProject1.cpp" "CMakeProject1.h")
target_link_libraries(CMakeProject1  ${OpenCV_LIBS} )
#target_link_libraries(CMakeProject1 /usr/local/lib/libopencv_viz.so)
