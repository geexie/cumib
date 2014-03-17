#
# The MIT License (MIT)
#
# Copyright (c) 2014 cuda.geek (cuda.geek@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

set(the_name cumib)
set(the_description "The CUDA (micro-)benchmarcs.")
project(${the_name})

set(${the_name}_MAJOR 0)
set(${the_name}_MINOR 1)
set(${the_name}_PATCH 0)
set(${the_name}_VERSION ${${the_name}_MAJOR}.${${the_name}_MINOR}.${${the_name}_PATCH})
add_definitions(-DCUMIB_VERSION=${${the_name}_VERSION})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_DEBUG_POSTFIX d)

include_directories("${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xptxas -dlcm=ca")
add_executable(${the_name}_global global_load.cu print_device_info.cu)