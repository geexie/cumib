/*
* The MIT License (MIT)
*
* Copyright (c) 2013 cuda.geek (cuda.geek@gmail.com)
*
* Permission is hereby granted, free of charge, to any  person obtaining a copy of
* this software  and associated  documentation  files (the "Software"), to deal in
* the Software without  restriction, including  without  limitation  the rights to
* use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
* the Software,  and to permit persons to  whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above  copyright notice  and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE  SOFTWARE IS  PROVIDED "AS IS",  WITHOUT  WARRANTY OF  ANY  KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR  PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS  BE LIABLE FOR ANY CLAIM,  DAMAGES OR OTHER LIABILITY, WHETHER
* IN  AN  ACTION  OF CONTRACT,  TORT OR OTHERWISE,  ARISING  FROM,  OUT  OF  OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "cudassert.cuh"
#include "flatten.cuh"

#include <stdint.h>
#include <cstdlib>
#include <stdio.h>

template<typename T>
struct Ldg
{
    __device__ __forceinline__ T* operator()(const T *ptr)
    {
#if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
        unsigned long long ret;
        asm volatile ("ld.global.nc.u64 %0, [%1];"  : "=l"(ret) : "l" (ptr));
        return reinterpret_cast<T*>((T)ret);
#else
        return (T*)0;
#endif
    }
};

template<typename T>
struct Ld
{
    __device__ __forceinline__ T* operator()(const T *j)
    {
        return *(T **)j;
    }
};

template<typename T>
struct Nop
{
    __device__ __forceinline__ T* operator()(T *j)
    {
        asm volatile ("" : : : "memory");
        return j;
    }
};

template<typename T>
struct TestPrint
{
    T* operator()(T* ptr) const
    {
        printf("%p\n", ptr);
        ++ptr;
        return ptr;
    }
};

static int getL2CacheSize(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.l2CacheSize;
}

static int getMajorCC(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.major;
}

static size_t getGmemSize(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.totalGlobalMem;
}

template<typename T, typename D, typename L>
__global__ void latency_kernel(T** a, int array_size, int stride, int inner_iterations, D* latency,
                               int declared_cache_size, bool do_warnup, L func)
{
    // disable unused lanes
    if (threadIdx.x >= stride) return;

    T *j = ((T*) a) + threadIdx.y * array_size +  threadIdx.x;

    int repeats = array_size / stride;

    D start_time, end_time;
    volatile D sum_time = 0;

    latency[0] = 0;
    int wurnup_repeats = min(repeats, declared_cache_size / stride / (int)sizeof(T));

    if (do_warnup)
    {
        for (int curr = 0; curr < repeats; ++curr)
            j = func(j);
    }

    // Yes, it's race conditions but we go not care.
    if (blockIdx.x > 64)
        ((T*)a)[array_size * blockDim.y + inner_iterations] = (T)j;

    for (int k = 0; k < inner_iterations; ++k)
    {
        T *j = ((T*) a) + threadIdx.y * array_size +  threadIdx.x;
        start_time = clock64();
        for (int curr = 0; curr < wurnup_repeats; ++curr)
            j = func(j);
        end_time = clock64();

        sum_time += (end_time - start_time);

        // the race condition again
        ((T*)a)[array_size * blockDim.y + k] = (T)j;
    }

    // now we need to store correct data. Ensure correct data in case of multi-warp block
    if (!threadIdx.x)
        atomicAdd((unsigned long long int*)latency, (unsigned long long int)sum_time);
}

struct CacheBenchmarker
{
    typedef long long int clock_64_t;

    CacheBenchmarker(int _stride, int _inner_iterations, int _outer_iterations, int _declared_cache_size, bool _do_warnup)
    : stride(_stride), inner_iterations(_inner_iterations), outer_iterations(_outer_iterations),
    declared_cache_size(_declared_cache_size), do_warnup(_do_warnup)
    {}

    template<typename T, typename L>
    void run_global_test(int array_size, int block_x, int block_y)
    {
        // allocate 1 more element to prevent compiler optimization that throws out kernel inner loop computations,
        // if kernel does not perform any result load.
        static const int extra_tail_size = (inner_iterations + 1);

        T*  h_a = (T *)malloc(sizeof(T) * block_y * (array_size + extra_tail_size));

        clock_64_t   h_latency   = 0;
        clock_64_t   sum_latency = 0;

        T ** d_a;
        clock_64_t* d_latency;

        cuda_assert(cudaMalloc((void **) &d_a, block_y * sizeof(T) * (array_size + extra_tail_size)));
        cuda_assert(cudaMalloc((void **) &d_latency, sizeof(clock_64_t)));


        for (int y = 0; y < block_y; ++y)
            for (int i = 0; i < array_size; ++i)
                h_a[y * array_size + i] = ((uintptr_t)d_a) + y * array_size * sizeof(T) + (i + stride) * sizeof(T);

        cuda_assert(cudaMemcpy((void *)d_a, (void *)h_a, sizeof(T) * array_size * block_y, cudaMemcpyHostToDevice));

        // ToDo: try better approaches to ensure that benchmark's kernel is only one executed.
        cuda_assert(cudaDeviceSynchronize());

        dim3 block = dim3(block_x, block_y);
        dim3 grid = dim3(1);

        L func;

        for (int outer = 0; outer < outer_iterations; ++outer)
        {
            latency_kernel<<<grid, block>>>(d_a, array_size, stride, inner_iterations, d_latency,
                                            declared_cache_size, do_warnup, func);

            cuda_assert(cudaDeviceSynchronize ());
            cuda_assert(cudaGetLastError());

            cuda_assert(cudaMemcpy((void *)&h_latency, (void *)d_latency, sizeof(clock_64_t), cudaMemcpyDeviceToHost));
            sum_latency += h_latency;
        }

        cuda_assert(cudaFree((void*)d_a));
        cuda_assert(cudaFree((void*)d_latency));

        free(h_a);

        printf("%lu, %d\n", array_size * sizeof(T) * block_y,
            static_cast<int>(sum_latency / ( min(array_size, declared_cache_size / (int)sizeof(T)) / (double)stride * outer_iterations * inner_iterations * block_y)));
    }

private:
    int stride;
    int inner_iterations;
    int outer_iterations;
    int declared_cache_size;
    bool do_warnup;
};

// usage:
// ./global_<arch> <device id> <cache type> <transaction width> <number of invocations> <do warn-up pass>
// warn-up fill the cache until it's size according to specification.
int main(int argc, char const *argv[])
{
    typedef uintptr_t test_type;
    static const int elem_size = sizeof(test_type);
    static const int min_array_size = 256;

    enum
    {
        DEVICE_ID_INDEX = 1,
        CACHE_TYPE_INDEX = 2,
        TRANSACTION_WIDTH_INDEX = 3,
        NUM_KERNEL_INV_INDEX = 4,
        DO_WARNUP_INDEX = 5
    };

    int deviceId = 0;
    if (argc > DEVICE_ID_INDEX)
        deviceId = atoi(argv[DEVICE_ID_INDEX]);
    printf("run benchmark on device #%d\n", deviceId);

    // retrieve current device compute capability
    const int cc_major = getMajorCC(deviceId);
    if (cc_major == 1)
    {
        printf("Global memory is uncached for Tesla generation devices.\n");
        return 0;
    }

    int cahce_type = 2;
    if (argc > CACHE_TYPE_INDEX)
        cahce_type = atoi(argv[CACHE_TYPE_INDEX]);

    // assume l2 transaction is 32 byte-wide with 4 sub-partitions.
    // This assumptions is truth for Kepler devices.
    // l1 transaction is always 128 byte-wide. Actual transaction optimization (shrinking 128-byte to 32 or 64 byte)
    // is performed in l2.
    int transactionWidth = 128;

    if (argc > TRANSACTION_WIDTH_INDEX)
        transactionWidth = atoi(argv[TRANSACTION_WIDTH_INDEX]);
    const int stride = transactionWidth / elem_size;

    // # of iterations inside the kernel and # of kernel invocations.
    const int innerIterations = 1;
    int outerIterations = 10;

    if (argc > NUM_KERNEL_INV_INDEX)
        outerIterations = atoi(argv[NUM_KERNEL_INV_INDEX]);

    // execute strided load function one time before main measurements
    bool doWarnup = true;
    if (argc > DO_WARNUP_INDEX)
        doWarnup = (bool)atoi(argv[DO_WARNUP_INDEX]);

    cuda_assert(cudaSetDevice(deviceId));
    printCudaDeviceInfo(deviceId);

    switch (cahce_type)
    {
        case 1:
        {
            // Fermi: it uses L1 for global loads. Lets use it
            if (cc_major == 2)
            {
                // configuration 16kb
                static const int l1_size = 16 * 1024;
                printf("configured for l1 size %d bytes\n", l1_size);

                const int declaredL2Size = getL2CacheSize(deviceId);
                const int maxCache = declaredL2Size + (declaredL2Size >> 2);

                cuda_assert(cudaFuncSetCacheConfig(latency_kernel<test_type, long long int, Ld<test_type> >,
                    cudaFuncCachePreferShared));
                CacheBenchmarker benchmarker(stride, innerIterations, outerIterations, l1_size, doWarnup);

                for (int N = min_array_size; N <= maxCache / elem_size; N += stride)
                    benchmarker.run_global_test<test_type, Ld<test_type> >(N, stride, 1);

            }
            // Kepler: try to benchmark LDG
            else
            {
                int block_x = 32;
                int block_y = 4;

                static const int dc_size = 16 * 1024;
                printf("texture cache size %d bytes\n", dc_size);

                const int declaredL2Size = getL2CacheSize(deviceId);
                const int maxCache  = 64 * 1024;

                CacheBenchmarker benchmarker(stride, innerIterations, outerIterations, dc_size, doWarnup);

                for (int N = min_array_size; N <= maxCache / elem_size; N += stride)
                    benchmarker.run_global_test<test_type, Ldg<test_type> >(N, block_x, block_y);
            }

            break;
        }
        case 2:
        {
            const int declaredL2Size = getL2CacheSize(deviceId);
            printf("declared l2 size %d bytes\n", declaredL2Size);

            // to be sure that all footprint will be covered.
            // for GTX650 with 256kb l2 it'll be 327680.
            const int maxCache = declaredL2Size + (declaredL2Size >> 2);

            CacheBenchmarker benchmarker(stride, innerIterations, outerIterations, declaredL2Size, doWarnup);

            for (int N = min_array_size; N <= maxCache / elem_size; N += stride)
                benchmarker.run_global_test<test_type, Ld<test_type> >(N, stride, 1);
            break;
        }
    }
    return 0;
}
