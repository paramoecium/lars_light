cmake_minimum_required (VERSION 2.8)
project (test_lars)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(mathOperations src/mathOperations.cpp)
add_library(lars src/lars.cpp)

set(CMAKE_C_COMPILER "/opt/intel/composer_xe_2013_sp1.1.106/bin/intel64/icc")
set(CMAKE_CXX_COMPILER "/opt/intel/composer_xe_2013_sp1.1.106/bin/intel64/icpc")

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -march=core-avx2")

set(test_source_files test.cpp)
add_executable(test_lars ${test_source_files})
target_link_libraries(test_lars lars)
target_link_libraries(test_lars mathOperations)


add_executable(measure measure.cpp)
target_link_libraries(measure lars)
target_link_libraries(measure mathOperations)
