/*
* The MIT License (MIT)
*
* Copyright (c) 2014 cuda.geek (cuda.geek@gmail.com)
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

#include "operations.cuh"
#include "cudassert.cuh"
#include <iostream>
#include <typeinfo>

using namespace cumib;
using namespace std;

template<class Op, typename T, int times> struct Repeat
{
    __device__ __forceinline__ T operator()(const Op& op, const T& a, const T& b) const
    {
        Repeat<Op, T, times-1> prev;
        T tmp = prev(op, a, b);
        return op(tmp, b);
    }
};

template<class Op, typename T> struct Repeat<Op, T, 1>
{
    __device__ __forceinline__ T operator()(const Op& op, const T& a, const T& b) const
    {
        return op(a, b);
    }
};

template<class Op, int times> struct RepeatControl
{
    __device__ __forceinline__ void operator()(const Op& op) const
    {
        RepeatControl<Op, times-1> prev; prev(op);
        return op();
    }
};

template<class Op> struct RepeatControl<Op, 1>
{
    __device__ __forceinline__ void operator()(const Op& op) const
    {
        return op();
    }
};

template<typename T, typename D, typename Op, int REPEATS>
static __global__ void latency_kernel(const T* in, T* out, D* latency, int inner_repeats)
{
    Op op;
    Repeat<Op, T, REPEATS> composite_op;

    T a = in[threadIdx.x];
    T b = in[threadIdx.x+1];

    D start = D(0), stop = D(0);

    start = clock64();

    for (int i=0; i<inner_repeats/REPEATS; ++i)
    {
        a = composite_op(op, a, b);
    }

    stop = clock64();

    out[threadIdx.x] = (a + b);

    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2]   = start;
    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop;
}

template<typename D, typename Op, int REPEATS>
static __global__ void latency_kernel_control(D* latency, int inner_repeats)
{
    Op op;
    RepeatControl<Op, REPEATS> composite_op;

    D start = D(0), stop = D(0);

    start = clock64();

    for (int i=0; i<inner_repeats/REPEATS; ++i)
    {
        composite_op(op);
    }

    stop = clock64();

    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2]   = start;
    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop;
}

template<typename T, typename D, int REPEATS>
struct LatencyTest
{
    template <typename Op>
    static void measure()
    {
        D latency_value[2];
        D latency_value_avg = D(0);


        T* src; cuda_assert(cudaMalloc(&src, sizeof(T)*2));
        T* dst; cuda_assert(cudaMalloc(&dst, sizeof(T)*2));

        D* latency; cuda_assert(cudaMalloc(&latency, sizeof(latency_value)));

        static const int inner_repeats = 1024;
        static const int outer_repeats = 2048;

        for (int i = 0; i < outer_repeats; ++i)
        {
            cuda_assert(cudaGetLastError());
            latency_kernel<T, D, Op, REPEATS><<<1, 1>>>(src, dst, latency, inner_repeats);
            cuda_assert(cudaGetLastError());
            cuda_assert(cudaThreadSynchronize());

            cudaMemcpy(&latency_value[0], latency, sizeof(latency_value), cudaMemcpyDeviceToHost);
            latency_value_avg += latency_value[1] - latency_value[0];
        }

        printf("%s %.3f clocks\n", TypeTraits<Op>::name() ,
         ((double)(latency_value_avg)/(inner_repeats*outer_repeats)));
    }
};

template<typename D, int REPEATS>
struct LatencyTestControl
{
    template <typename Op>
    static void measure()
    {
        D latency_value[2];
        D latency_value_avg = D(0);

        D* latency; cuda_assert(cudaMalloc(&latency, sizeof(latency_value)));

        static const int inner_repeats = 1024;
        static const int outer_repeats = 2048;

        for (int i = 0; i < outer_repeats; ++i)
        {
            cuda_assert(cudaGetLastError());
            latency_kernel_control<D, Op, REPEATS><<<1, 1>>>(latency, inner_repeats);
            cuda_assert(cudaGetLastError());
            cuda_assert(cudaThreadSynchronize());

            cudaMemcpy(&latency_value[0], latency, sizeof(latency_value), cudaMemcpyDeviceToHost);
            latency_value_avg += latency_value[1] - latency_value[0];
        }

        printf("%s %.3f clocks\n", TypeTraits<Op>::name() ,
         ((double)(latency_value_avg)/(inner_repeats*outer_repeats)));
    }
};

template<typename T>
void run_all_latency_tests()
{
    typedef typename ConstructOperationList<Add<T>, Sub<T>, Mul<T>, Div<T>, Mad<T>, Min<T> >::OpList BaseMathList;
    ForEach<BaseMathList, LatencyTest<T, long long int, 128> > all;
    all();
}

template<typename T>
void run_type_agnostic_tests()
{
    typedef typename ConstructOperationList<Shfl<T>, Ballot<T>, All<T>, Any<T> >::OpList BaseMathList;
    ForEach<BaseMathList, LatencyTest<T, long long int, 128> > all;
    all();
}

void run_control_tests()
{
    typedef typename ConstructOperationList<Sync >::OpList BaseMathList;
    ForEach<BaseMathList, LatencyTestControl<long long int, 128> > all;
    all();
}

int main(int argc, char const *argv[])
{
    run_all_latency_tests<int>();
    printf("\n");

    run_all_latency_tests<unsigned int>();
    printf("\n");

    run_all_latency_tests<float>();
    printf("\n");

    run_all_latency_tests<long long int>();
    printf("\n");

    run_all_latency_tests<double>();
    printf("\n");

    run_type_agnostic_tests<int>();
    printf("\n");

    run_control_tests();
    return 0;
}