/*
* The MIT License (MIT)
*
* Copyright (c) 2013-2014 cuda.geek (cuda.geek@gmail.com)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of
* this software and associated documentation files (the "Software"), to deal in
* the Software without restriction, including without limitation the rights to
* use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
* the Software, and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// simple function that print information about device which is currently benchmarked.
// Inspired by deviceQuery CUDA SDK example. see http://docs.nvidia.com/cuda/cuda-samples/index.html#cudalibraries

#include "cudassert.cuh"

void printCudaDeviceInfo(int dev, bool verbose)
{
    int driverVersion = 0, runtimeVersion = 0;

    cuda_assert( cudaDriverGetVersion(&driverVersion) );
    cuda_assert( cudaRuntimeGetVersion(&runtimeVersion) );

    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
        driverVersion / 1000, driverVersion % 100, runtimeVersion / 1000, runtimeVersion % 100);

    const char *computeMode[] = {
        "Default",
        "Exclusive",
        "Prohibited",
        "Exclusive Process",
        "Unknown",
        0
    };

    {
        cudaDeviceProp prop;
        cuda_assert( cudaGetDeviceProperties(&prop, dev) );

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);
        printf("  Compute capability (#SM):                      %d.%d (%d)\n", prop.major, prop.minor, prop.multiProcessorCount);

        if (verbose)
            printf("  Warp size:                                     %d\n", prop.warpSize);

        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);
        printf("  Memory Clock Speed:                            %.2f GHz\n", prop.memoryClockRate * 1e-6f);
        printf("  Memory Bus width  :                            %d Bit\n", prop.memoryBusWidth);

        printf("  Device supports caching globals in L1          %s\n", prop.globalL1CacheSupported ? "Yes" : "No");
        printf("  Device supports caching locals in L1           %s\n", prop.localL1CacheSupported ? "Yes" : "No");
        printf("  Device supports allocating managed memory      %s\n", prop.managedMemory ? "Yes" : "No");
        printf("  Device supports stream priorities              %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");

        if (verbose)
        {
            printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)prop.totalGlobalMem/1048576.0f, (unsigned long long) prop.totalGlobalMem);
            printf("  Size of L2 cache in Kb (bytes) :               %d Kb (%u bytes)\n", (int)prop.l2CacheSize/1024, (int)prop.l2CacheSize);
            printf("  Total amount of constant memory:               %u bytes\n", (int)prop.totalConstMem);
            printf("  Total amount of shared memory per block:       %u bytes\n", (int)prop.sharedMemPerBlock);
            printf("  Total amount of shared memory per SM   :       %u bytes\n", (int)prop.sharedMemPerMultiprocessor);
            printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
            printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
            printf("  Maximum number of threads per SM:              %d\n", prop.maxThreadsPerMultiProcessor);
            printf("  Maximum memory pitch:                          %u bytes\n", (int)prop.memPitch);
            printf("  Texture alignment:                             %u bytes\n", (int)prop.textureAlignment);
            printf("  Texture pitch alignment:                       %u bytes\n", (int)prop.texturePitchAlignment);
        }

        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n",
            (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("  Concurrent kernel execution:                   %s\n", prop.concurrentKernels ? "Yes" : "No");

        if (verbose)
        {
            printf("  Device has ECC support enabled:                %s\n", prop.ECCEnabled ? "Yes" : "No");
            printf("  Device is using TCC driver mode:               %s\n", prop.tccDriver ? "Yes" : "No");
        }

        printf("  Device supports Unified Addressing (UVA):      %s\n", prop.unifiedAddressing ? "Yes" : "No");

        if (verbose)
        {
            printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", prop.pciBusID, prop.pciDeviceID );
            printf("  Compute Mode:\n");
        }
        printf("  Compute mode:                                  %s \n", computeMode[prop.computeMode]);
    }
    fflush(stdout);
}
