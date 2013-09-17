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

#include <stdio.h>

typedef int test_t;

static __device__ __forceinline__ unsigned int laneId()
{
    unsigned int ret;
#ifdef __CUDA_ARCH__
    asm("mov.u32 %0, %laneid;" : "=r"(ret) );
#endif
    return ret;
}

__global__ void laneid_kernel(test_t* src)
{
    src[(blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x + threadIdx.x) * 4 + 0] = threadIdx.z;
    src[(blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x + threadIdx.x) * 4 + 1] = threadIdx.y;
    src[(blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x + threadIdx.x) * 4 + 2] = threadIdx.x;
    src[(blockDim.x * blockDim.y * threadIdx.z + threadIdx.y * blockDim.x + threadIdx.x) * 4 + 3] = laneId();
}

struct LaneIdBenchmarker
{
    LaneIdBenchmarker() {}

    void run_laneid_test(const dim3 block)
    {
        const int array_size = block.x * block.y * block.z * 4;
        test_t* h_ptr = new test_t[array_size];
        test_t* d_ptr;

        cuda_assert(cudaMalloc((void**)&d_ptr, sizeof(test_t) * array_size));

        laneid_kernel<<<1, block>>>(d_ptr);
        cuda_assert(cudaDeviceSynchronize());
        cuda_assert(cudaGetLastError());

        cuda_assert(cudaMemcpy((void*)h_ptr, (void*)d_ptr, sizeof(test_t) * array_size, cudaMemcpyDeviceToHost));

        printf("results for block (%d, %d, %d) \n", block.x, block.y, block.z);

        printf("tid.z\ttid.y\ttid.x\tlane\n");
        for (int i = 0; i < array_size; i += 4)
        {
            printf("%d\t%d\t%d\t%d\n", h_ptr[i + 0], h_ptr[i + 1], h_ptr[i + 2], h_ptr[i + 3]);
        }

        cuda_assert(cudaFree((void*)d_ptr));
        delete[] h_ptr;
    }
};


int main()
{
    LaneIdBenchmarker benchmark;
    benchmark.run_laneid_test(dim3(16, 2, 1));
    benchmark.run_laneid_test(dim3(19, 2, 1));
    benchmark.run_laneid_test(dim3(16, 2, 2));
    return 0;
}