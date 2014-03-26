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

using namespace cumib;
using namespace std;


template<class Op, typename T, int times> struct Repeat
{
    void operator()(Op op, const T& a, const T& b) const
    {
        Repeat<Op, T, times-1> prev;
        return prev(op, a, op(a, b));
    }
};

template<class Op, typename T> struct Repeat<Op, T, 1>
{
    void operator()(Op op, const T& a, const T& b) const
    {
        return op(a, b);
    }
};

template<typename T, typename D, typename Op>
static __global__ void latency_kernel(const T* in, T* out, D* latency, int inner_repeats)
{
    Op op;
    Repeat<Op, T, 64> composite_op;

    T a = in[threadIdx.x];
    T b = in[threadIdx.x+1];

    D start = D(0), stop = D(0);

    start = clock64();

    for (int i=0; i<inner_repeats/64; ++i)
    {
        composite_op(a, b);
    }

    start = clock64();

    out[threadIdx] = (a + b);

    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2]   = start;
    latency[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop;

};

template<typename T, typename D, typename Op>
struct LatencyTest
{
    static void measure()
    {
        std::cout << "measure" << std::endl;
        // T* src;
        // T* dst;
        // D* latency;
        // D latency_value[2];
        // D latency_value_avg = D(0);

        // static const int inner_repeats = 256;
        // static const int outer_repeats =  16;

        // for (int i = 0; i < outer_repeats; ++i)
        // {
        //     latency_kernel<T, D, Op><<<1, 1>>>(src, dst, latency);
        //     cuda_assert(cudaGetLastError());
        //     cuda_assert(cudaThreadSynchronize());

        //     cudaMemcpy(latency, &latency_value, sizeof(latency_value), cudaMemcpyDeviceToHost);
        //     latency_value_avg += latency_value[1] - latency_value[0];
        // }
        // printf ("%.3f clock\n", ((double)(latency_value_avg)/(inner_repeats*outer_repeats)));
    }
};

int main(int argc, char const *argv[])
{
    typedef typename ConstructOperationList<And<int>, Sub<int> >::OpList BaseMathList;

    ForEach<BaseMathList/*, LatencyTest*/> all;
    all();

    return 0;
}