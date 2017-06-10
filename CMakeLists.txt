cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE OFF)
project(lars_light)

# ================================================================================
# Reset the C/C++ flags, and take full control
# ================================================================================
set(CMAKE_C_FLAGS           "")
set(CMAKE_CXX_FLAGS         "")
set(CMAKE_C_FLAGS_RELEASE   "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_DEBUG     "")
set(CMAKE_CXX_FLAGS_DEBUG   "")

# ================================================================================
# C++ Global Flags
# ================================================================================
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    # clang - C++ flags
    set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -mavx2 -mfma")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    # GNU GCC - C++ flags
    set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

    # Intel ICC - C++ flags
    #set(CMAKE_CXX_FLAGS "-O3 -ansi-alias -no-vec -unroll=0 ")
    set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -xHost -ansi-alias -march=core-avx2 -vec_report1")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

    # Microsoft Visual Studio - C++ flags
    set(CMAKE_CXX_FLAGS "-O2 -std=c++11")

endif()

# ================================================================================
# C Global Flags
# ================================================================================

if (CMAKE_C_COMPILER_ID MATCHES "Clang")

    # clang - CC flags
    set(CMAKE_C_FLAGS "-O3 -march=core-avx2 -mavx2 -mfma")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

    # GNU GCC - CC flags
    set(CMAKE_C_FLAGS "-O3 -march=core-avx2")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")

    # Intel ICC - CC flags
    #set(CMAKE_C_FLAGS "-O3 -ansi-alias -no-vec -unroll=0 ")
    set(CMAKE_C_FLAGS "-O3 -std=c++11 -xHost -ansi-alias -march=core-avx2 -vec_report1")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")

    # Microsoft Visual Studio - CC flags
    set(CMAKE_C_FLAGS "-O2 -std=c++11")

endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(mathOperations src/mathOperations.cpp)
add_library(lars src/lars.cpp)

set(test_source_files test.cpp)
add_executable(test_lars ${test_source_files})
target_link_libraries(test_lars lars)
target_link_libraries(test_lars mathOperations)

add_executable(measure measure.cpp)
target_link_libraries(measure lars)
target_link_libraries(measure mathOperations)
