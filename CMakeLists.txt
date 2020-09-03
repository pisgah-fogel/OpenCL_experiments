#Minimal OpenCL CMakeLists.txt by StreamHPC

cmake_minimum_required (VERSION 3.1)

project(GreatProject)

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable (main main.cpp)
target_include_directories (main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (main ${OpenCL_LIBRARY})
#/usr/lib/x86_64-linux-gnu/libOpenCL.so
