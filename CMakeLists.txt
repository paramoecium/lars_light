cmake_minimum_required (VERSION 2.8)
project (test_lars)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(mathOperations src/mathOperations.cpp)
add_library(lars src/lars.cpp)

set(CMAKE_CXX_FLAGS "-O3 -mavx -mfma")

set(test_source_files test.cpp)
add_executable(test_lars ${test_source_files})
target_link_libraries(test_lars lars)
target_link_libraries(test_lars mathOperations)


add_executable(measure measure.cpp)
target_link_libraries(measure lars)
target_link_libraries(measure mathOperations)
