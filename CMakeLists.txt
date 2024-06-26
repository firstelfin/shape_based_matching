cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)

OPTION(USE_PYBIND "Enable pybind" ON)

# debug or release
SET(CMAKE_BUILD_TYPE "Release")


# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CUDA_DIR "/usr/local/cuda/")
SET(pybind11_DIR "/data1/Installer/miniconda3InstallDir/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
SET(CMAKE_MODULE_PATH "/data1/Installer/miniconda3InstallDir/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

file(GLOB_RECURSE LIB_OPENCV /usr/local/lib/libopencv_*.so)
# opencv
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /usr/local/)
find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4 /data1/Installer/miniconda3InstallDir/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})

# test exe
add_executable(${PROJECT_NAME}_infer line2Dup.cpp SBM.cpp)
target_link_libraries(${PROJECT_NAME}_infer ${OpenCV_LIBS})

# pybind
if (USE_PYBIND)
    find_package(PythonLibsNew 3 REQUIRED)
    find_package(pybind11 REQUIRED)
    include(pybind11/Cmake/FindNumPy.cmake)
    set(SOURCE_PYBIND
        pybind11/pybind11.cpp
        pybind11/np2mat/ndarray_converter.cpp)
    include_directories(${INCLUDE_DIRECTORIES}
            ${PYTHON_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR})
    pybind11_add_module(${PROJECT_NAME}_py line2Dup.cpp ${SOURCE_PYBIND})
    target_link_libraries(${PROJECT_NAME}_py PRIVATE ${OpenCV_LIBS})
endif()