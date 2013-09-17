/*
* The MIT License (MIT)
*
* Copyright (c) 2013 cuda.geek (cuda.geek@gmail.com)
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

#include "cudassert.cuh"

#include <stdint.h>
#include <cstdlib>
#include <stdio.h>

int getL2CacheSize(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.l2CacheSize;
}

size_t getGmemSize(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.totalGlobalMem;
}

template<typename T, typename D>
__global__ void latency_kernel(T** a, int array_size, int stride, int inner_iterations, D* latency,
                               int declared_l2_size, bool do_warnup)
{
    T *j = (T*) a;

    int repeats = array_size / stride;

    D start_time, end_time;
    volatile D sum_time = 0;

    latency[0] = 0;
    int wurnup_repeats = min(repeats, declared_l2_size / stride);

    if (do_warnup)
    {
        for (int curr = 0; curr < wurnup_repeats; ++curr)
            j = *(T **)j;
    }

    ((T*)a)[array_size + inner_iterations] = (T)j;

    for (int k = 0; k < inner_iterations; ++k)
    {
        T *j = (T*) a;
        start_time = clock64();
        for (int curr = 0; curr < repeats; ++curr)
            j = *(T **)j;
        end_time = clock64();

        sum_time += (end_time - start_time);
        ((T*)a)[array_size + k] = (T)j;
    }

    latency[0] = sum_time;
}

struct CacheBenchmarker
{
    typedef long long int clock_64_t;

    CacheBenchmarker(int _stride, int _inner_iterations, int _outer_iterations, int _declared_l2_size, bool _do_warnup)
    : stride(_stride), inner_iterations(_inner_iterations), outer_iterations(_outer_iterations),
    declared_l2_size(_declared_l2_size), do_warnup(_do_warnup)
    {}

    template<typename T>
    void run_global_test(int array_size)
    {
        // allocate 1 more element to prevent compiler optimization that throws out kernel inner loop computations.
        static const int extra_tail_size = inner_iterations + 1;

        T*  h_a = (T *)malloc(sizeof(T) * (array_size + extra_tail_size));

        clock_64_t   h_latency   = 0;
        clock_64_t   sum_latency = 0;

        T ** d_a;
        clock_64_t * d_latency;

        cuda_assert(cudaMalloc((void **) &d_a, sizeof(T) * (array_size + extra_tail_size)));
        cuda_assert(cudaMalloc((void **) &d_latency, sizeof(clock_64_t)));

        for (int i = 0; i < array_size; i += stride)
            h_a[i] = ((uintptr_t)d_a) + ((i + stride) % array_size) * sizeof(T);

        cuda_assert(cudaMemcpy((void *)d_a, (void *)h_a, sizeof(T) * array_size, cudaMemcpyHostToDevice));

        // try better approaches to ensure that benchmark's kernel is only one executed.
        cuda_assert(cudaDeviceSynchronize());

        dim3 block = dim3(1);
        dim3 grid = dim3(1);

        for (int outer = 0; outer < outer_iterations; ++outer)
        {
            latency_kernel<<<grid, block>>>(d_a, array_size, stride, inner_iterations, d_latency,
                                            declared_l2_size, do_warnup);

            cuda_assert(cudaDeviceSynchronize ());
            cuda_assert(cudaGetLastError());

            cuda_assert(cudaMemcpy((void *)&h_latency, (void *)d_latency, sizeof(clock_64_t), cudaMemcpyDeviceToHost));
            sum_latency += h_latency;
        }

        cuda_assert(cudaFree((void*)d_a));
        cuda_assert(cudaFree((void*)d_latency));

        free(h_a);

        printf("%lu, %d\n", array_size * sizeof(T),
            static_cast<int>(sum_latency / (array_size / (double)stride * outer_iterations * inner_iterations)));
    }

private:
    int stride;
    int inner_iterations;
    int outer_iterations;
    int declared_l2_size;
    bool do_warnup;
};


int main(int argc, char const *argv[])
{
    typedef uintptr_t test_type;
    static const int elem_size = sizeof(test_type);
    static const int min_array_size = 256;

    int deviceId = 0;

    if (argc >= 2)
        deviceId = atoi(argv[1]);

    // assume l2 transaction is 32 byte-wide with 4 sub-partitions.
    // This assumptions is truth for kepler devices.
    int transactionWidth = 128;

    if (argc >= 3)
        transactionWidth = atoi(argv[2]);

    const int stride = transactionWidth / elem_size;

    printf("run benchmark on device #%d\n", deviceId);

    cuda_assert(cudaSetDevice(deviceId));
    printCudaDeviceInfo(deviceId);

    const int declaredL2Size = getL2CacheSize(deviceId);
    printf("declared l2 size %d bytes\n", declaredL2Size);

    // to be sure that all footprint will be covered.
    // for GTX650 with 256kb l2 it'll be 327680.
    const int maxCache = declaredL2Size + (declaredL2Size >> 2);

    // # of iterations inside the kernel and # of kernel invocations.
    const int innerIterations = 1;
    int outerIterations = 10;

    if (argc >= 4)
        outerIterations = atoi(argv[3]);

    // execute strided load function one time before main measurements
    bool doWarnup = false;

    if (argc >= 5)
        doWarnup = (bool)atoi(argv[4]);

    CacheBenchmarker benchmarker(stride, innerIterations, outerIterations, declaredL2Size, doWarnup);

    for (int N = min_array_size; N <= maxCache / elem_size; N += stride)
        benchmarker.run_global_test<test_type>(N);

    return 0;
}
