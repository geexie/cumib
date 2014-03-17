/*
* The MIT License (MIT)
*
* Copyright (c) 2013-2014 cuda.geek (cuda.geek@gmail.com)
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
#include "cumib.cuh"

#include <cmath>

using namespace cumib;

__global__ void inc_kernel(unsigned int *cnt)
{
    printf("w %d %u\n", threadIdx.x, *cnt);
    atomicAdd(cnt, 1);
    printf("b %d %u\n", threadIdx.x, *cnt);
}

template<typename T>
__global__ void copy1d1d(const T *src, T *dst)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
#else
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    dst[index] = src[index];
}

template<typename T>
__global__ void fillWithTid(T *dst)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] = index;
}

enum
{
    GMEM_DEVICE = 1,
    GMEM_HOST   = 2
};

template<int TYPE> struct Dispatcher;
template<> struct Dispatcher<GMEM_HOST> {};
template<> struct Dispatcher<GMEM_DEVICE> {};

template<typename T, typename D> struct MPtr
{
    MPtr(const size_t size, bool fill = false)
    {
        cuda_assert(cudaMalloc((void**) &_ptr, sizeof(T) * size));
        if (0 && fill)
        {
            fillWithTid<<<size / CTA_SIZE, CTA_SIZE>>>(_ptr);
            cuda_assert(cudaDeviceSynchronize());
            cuda_assert(cudaGetLastError());
        }
    }

    T* ptr() const {return _ptr;}

    const char* space() {return "gpu";}

    ~MPtr()
    {
        cuda_assert(cudaFree((void*) _ptr));
    }

private:
    T* _ptr;

    enum name {CTA_SIZE = 128};
};

template<typename T> struct MPtr<T, Dispatcher<GMEM_HOST> >
{
    MPtr(const size_t size, bool fill = false)
    {
        // cudaHostAllocMapped
        cuda_assert(cudaHostAlloc((void**) &_ptr, sizeof(T) * size, cudaHostAllocDefault));
        if (fill)
        {
            for (size_t i = 0; i < size; ++i)
                _ptr[i] = i;
        }
    }

    T* ptr() const {return _ptr;}

    const char* space() {return "host";}

    ~MPtr()
    {
        cuda_assert(cudaFreeHost((void*) _ptr));
    }

private:
    T* _ptr;
};

static int getMajorCC(int deviceId)
{
    cudaDeviceProp prop;
    cuda_assert( cudaGetDeviceProperties(&prop, deviceId) );
    // in bytes
    return prop.major;
}

static dim3 declareGrid(int size, int cta_size)
{
    int deviceId = 0;
    cuda_assert(cudaGetDevice(&deviceId));
    const int major_cc = getMajorCC(deviceId);

    dim3 grid(size/cta_size);
    if((major_cc < 3) && (size/cta_size) >= 65536)
        grid = dim3( static_cast<int>(sqrt(size/cta_size)), static_cast<int>(sqrt(size/cta_size)));

    return grid;
}

template<typename Pi, typename Po>
static void run_mapped_test(const size_t min_array_size, const size_t  max_array_size)
{
    Pi src(max_array_size, true);
    Po dst(max_array_size);

    HostTimer timer;

    static const int cta_size = 128;

    for (size_t size = min_array_size; size < max_array_size; size *=2)
    {
        printf("run: %zi\t", size);

        dim3 grid = declareGrid(size, cta_size);

        timer.go();
        copy1d1d<<<grid, cta_size>>>(src.ptr(), dst.ptr());
        cuda_assert(cudaDeviceSynchronize());
        cuda_assert(cudaGetLastError());

        float mtime = timer.measure();
        printf("%s -> %s:\t %f ms\n", src.space(), dst.space(), mtime);
    }
}

template<typename T>
static void run_mapped_tests(const size_t min_array_size, const size_t  max_array_size)
{
    run_mapped_test<MPtr<T,Dispatcher<GMEM_DEVICE> >, MPtr<T,Dispatcher<GMEM_DEVICE> > >(min_array_size, max_array_size);
    run_mapped_test<MPtr<T,Dispatcher<GMEM_DEVICE> >, MPtr<T,Dispatcher<GMEM_HOST> > >(min_array_size, max_array_size);
    run_mapped_test<MPtr<T,Dispatcher<GMEM_HOST> >,   MPtr<T,Dispatcher<GMEM_DEVICE> > >(min_array_size, max_array_size);
    run_mapped_test<MPtr<T,Dispatcher<GMEM_HOST> >,   MPtr<T,Dispatcher<GMEM_HOST> > >(min_array_size, max_array_size);
}

int main(int argc, char **argv)
{
    typedef unsigned int test_t;

    int deviceId = 0;

    if (argc >= 2)
        deviceId = atoi(argv[1]);

    cuda_assert(cudaSetDevice(deviceId));
    printCudaDeviceInfo(deviceId);

    static const size_t min_array_size = static_cast<size_t>(std::pow(2., 15.));
    static const size_t max_array_size = static_cast<size_t>(std::pow(2., 27));
    run_mapped_tests<test_t>(min_array_size, max_array_size);
}