#--------------------------------------------------------------
# default build-type (release)
# (the next lines must be called bevore project(seglib))
#--------------------------------------------------------------
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()


#--------------------------------------------------------------
# CMAKE settings
#--------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)


#--------------------------------------------------------------
# Project setting and versioning
#--------------------------------------------------------------
project(seglib)
set (CGP2D_VERSION_MAJOR 0)
set (CGP2D_VERSION_MAJOR 1)
set (OPENGM_VERSION_PATCH 0)


#--------------------------------------------------------------
# Find external packges
# - Python interpreter
# - boost python
# - VIGRA
# 
#--------------------------------------------------------------
FIND_PACKAGE(PythonInterp)
find_package(Boost COMPONENTS python REQUIRED)
find_package(VIGRA REQUIRED)
find_package (PythonLibs REQUIRED)


#--------------------------------------------------------------
# global headers
#--------------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
include_directories(include)
include_directories(include)



include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIR}
    ${VIGRA_INCLUDE_DIR}
    ${PYTHON_INCLUDE_PATH}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cgp2d
)
add_definitions(-std=c++0x)
#--------------------------------------------------------------
# add subdirs
# -src
#   - python
#   - unitttes
#--------------------------------------------------------------
add_subdirectory(src)