#
# The MIT License (MIT)
#
# Copyright (c) 2013 cuda.geek (cuda.geek@gmail.com)
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

##################################################################################
#   This is a simple make file that is not support cross-compilation yet.
# To cross-compile cumib override compiler and flag variables.
#
#   This makefile require CUDA compute capability greater of equal then 2.0 becau-
# se of device linking required for incremental build.
#
# This makefile works with cuda 5.5 with the target folder
##################################################################################

HOST_NAME = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
HOST_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

CC = g++
HOST_CC=$(CC)

CFLAGS=
LDFLAGS=

CUDA_ROOT ?= /usr/local/cuda

CUDA_LIB_DIR = $(CUDA_ROOT)/targets/$(HOST_ARCH)-$(HOST_NAME)/lib

ifeq ($(HOST_ARCH),armv7l)
CUDA_LIB_DIR = $(CUDA_ROOT)/targets/armv7-$(HOST_NAME)-gnueabihf/lib
endif

CUDA_RT = cudart
DEVICE_CC = $(CUDA_ROOT)/bin/nvcc -ccbin $(HOST_CC)

CUDA_GENERATION = 30

NVCC_FLAGS ?=
NVCC_FLAGS += -gencode arch=compute_$(CUDA_GENERATION),code=sm_$(CUDA_GENERATION) -Xptxas -dlcm=cg --verbose

################################################################################
# Targets
################################################################################

global: global_load_gevice.o print_device_info.o global_load.o
	$(HOST_CC) print_device_info.o global_load.o global_load_gevice.o -L $(CUDA_LIB_DIR) -l$(CUDA_RT) -o global

global_load_gevice.o: global_load.o print_device_info.o
	$(DEVICE_CC) $(NVCC_FLAGS) -dlink global_load.o print_device_info.o -o global_load_gevice.o

global_load.o: global_load.cu cudassert.cuh
	$(DEVICE_CC) $(NVCC_FLAGS) -c --relocatable-device-code=true  global_load.cu -o global_load.o

print_device_info.o: print_device_info.cu cudassert.cuh
	$(DEVICE_CC) $(NVCC_FLAGS) -c --relocatable-device-code=true  print_device_info.cu -o print_device_info.o

laneid: laneid.cu cudassert.cuh
	$(DEVICE_CC) $(NVCC_FLAGS) laneid.cu -o laneid

all: global laneid

.PHONY: clean

clean:
	rm *.o global
