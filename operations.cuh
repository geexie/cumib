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

#ifndef __CUMIB_OPERATIONS_CUH__
#define __CUMIB_OPERATIONS_CUH__

#include <iostream>

namespace cumib {

// minimum & maximum
template<typename T>
struct Min
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return min(a,b); }
};

template<>
struct Min<int>
{
    __device__ __forceinline__ int operator()(const int& a, const int& b) const
    { int tmp; asm volatile ("min.s32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Min<long long int>
{
    __device__ __forceinline__ long long int operator()(const long long int& a, const long long int& b) const
    { long long int tmp; asm volatile ("min.s64 %0, %1, %2;": "=l"(tmp):"l"(a), "l"(b)); return tmp; }
};

template<>
struct Min<unsigned int>
{
    __device__ __forceinline__ unsigned int operator()(const unsigned int& a, const unsigned int& b) const
    { unsigned int tmp; asm volatile ("min.u32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Min<float>
{
    __device__ __forceinline__ float operator()(const float& a, const float& b) const
    { float tmp; asm volatile ("min.f32 %0, %1, %2;": "=f"(tmp):"f"(a), "f"(b)); return tmp; }
};

template<>
struct Min<double>
{
    __device__ __forceinline__ double operator()(const double& a, const double& b) const
    { double tmp; asm volatile ("min.f64 %0, %1, %2;": "=d"(tmp):"d"(a), "d"(b)); return tmp; }
};

// multiply & addition
template<typename T>
struct Mad
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a*b+b; }
};

// addition
template<typename T>
struct Add
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a+b; }
};

template<>
struct Add<int>
{
    __device__ __forceinline__ int operator()(const int& a, const int& b) const
    { int tmp; asm volatile ("add.s32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Add<long long int>
{
    __device__ __forceinline__ long long int operator()(const long long int& a, const long long int& b) const
    { long long int tmp; asm volatile ("add.s64 %0, %1, %2;": "=l"(tmp):"l"(a), "l"(b)); return tmp; }
};

template<>
struct Add<unsigned int>
{
    __device__ __forceinline__ unsigned int operator()(const unsigned int& a, const unsigned int& b) const
    { unsigned int tmp; asm volatile ("add.u32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Add<float>
{
    __device__ __forceinline__ float operator()(const float& a, const float& b) const
    { float tmp; asm volatile ("add.f32 %0, %1, %2;": "=f"(tmp):"f"(a), "f"(b)); return tmp; }
};

template<>
struct Add<double>
{
    __device__ __forceinline__ double operator()(const double& a, const double& b) const
    { double tmp; asm volatile ("add.f64 %0, %1, %2;": "=d"(tmp):"d"(a), "d"(b)); return tmp; }
};

// subtraction
template<typename T>
struct Sub
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a-b; }
};

template<>
struct Sub<int>
{
    __device__ __forceinline__ int operator()(const int& a, const int& b) const
    { int tmp; asm volatile ("sub.s32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Sub<unsigned int>
{
    __device__ __forceinline__ unsigned int operator()(const unsigned int& a, const unsigned int& b) const
    { unsigned int tmp; asm volatile ("sub.u32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct Sub<float>
{
    __device__ __forceinline__ float operator()(const float& a, const float& b) const
    { float tmp; asm volatile ("sub.f32 %0, %1, %2;": "=f"(tmp):"f"(a), "f"(b)); return tmp; }
};

template<>
struct Sub<long long int>
{
    __device__ __forceinline__ long long int operator()(const long long int& a, const long long int& b) const
    { long long int tmp; asm volatile ("sub.s64 %0, %1, %2;": "=l"(tmp):"l"(a), "l"(b)); return tmp; }
};

template<>
struct Sub<double>
{
    __device__ __forceinline__ double operator()(const double& a, const double& b) const
    { double tmp; asm volatile ("sub.f64 %0, %1, %2;": "=d"(tmp):"d"(a), "d"(b)); return tmp; }
};


// multiplication
template<typename T>
struct Mul
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a*b; }
};

// division
template<typename T>
struct Div
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a/b; }
};

// shuffle
template<typename T>
struct Shfl
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return __shfl(a, b); }
};

// vote
template<typename T>
struct Ballot
{
    __device__ __forceinline__ T operator()(const T& a, const T& /*b*/) const { return __ballot(a); }
};

template<typename T>
struct All
{
    __device__ __forceinline__ T operator()(const T& a, const T& /*b*/) const { return __all(a); }
};

template<typename T>
struct Any
{
    __device__ __forceinline__ T operator()(const T& a, const T& /*b*/) const { return __any(a); }
};

struct Sync
{
    __device__ __forceinline__ void operator()(void) const { return __syncthreads(); }
};

struct Prmt
{
    __device__ __forceinline__ int operator()(const int& a, const int& b) const
    { int tmp; asm volatile ("prmt.b32 %0, %1, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

struct Empty {};

template<typename Op1, typename Op2>
struct OperationList
{
    typedef Op1 Head;
    typedef Op2 Tail;
};

// construct operation list
template<
    typename Op1  = Empty,typename Op2  = Empty,typename Op3  = Empty,typename Op4  = Empty,typename Op5  = Empty,
    typename Op6  = Empty,typename Op7  = Empty,typename Op8  = Empty,typename Op9  = Empty,typename Op10 = Empty,
    typename Op11 = Empty,typename Op12 = Empty,typename Op13 = Empty,typename Op14 = Empty,typename Op15 = Empty,
    typename Op16 = Empty,typename Op17 = Empty,typename Op18 = Empty,typename Op19 = Empty,typename Op20 = Empty,
    typename Op21 = Empty,typename Op22 = Empty,typename Op23 = Empty,typename Op24 = Empty,typename Op25 = Empty,
    typename Op26 = Empty,typename Op27 = Empty,typename Op28 = Empty,typename Op29 = Empty,typename Op30 = Empty,
    typename Op31 = Empty,typename Op32 = Empty,typename Op33 = Empty,typename Op34 = Empty,typename Op35 = Empty,
    typename Op36 = Empty,typename Op37 = Empty,typename Op38 = Empty,typename Op39 = Empty,typename Op40 = Empty,
    typename Op41 = Empty,typename Op42 = Empty,typename Op43 = Empty,typename Op44 = Empty,typename Op45 = Empty,
    typename Op46 = Empty,typename Op47 = Empty,typename Op48 = Empty,typename Op49 = Empty,typename Op50 = Empty
>
struct ConstructOperationList
{
    typedef typename ConstructOperationList<      Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9,Op10,
                                   Op11,Op12,Op13,Op14,Op15,Op16,Op17,Op18,Op19,Op20,
                                   Op21,Op22,Op23,Op24,Op25,Op26,Op27,Op28,Op29,Op30,
                                   Op31,Op32,Op33,Op34,Op35,Op36,Op37,Op38,Op39,Op40,
                                   Op41,Op42,Op43,Op44,Op45,Op46,Op47,Op48,Op49,Op50>::OpList OpListTail;

    typedef OperationList<Op1, OpListTail> OpList;
};

template<>
struct ConstructOperationList<>
{
    typedef Empty OpList;
};

// for each operation in list apply
template <class OpList, class OpRoot> struct ForEach;

template <class OpRoot> struct ForEach<Empty, OpRoot>
{
    void operator()(){}
};

template <class T, class U, class OpRoot>
struct ForEach< OperationList<T, U>, OpRoot>
{
    void operator()()
    {
        OpRoot::template measure<T>();
        ForEach<U, OpRoot> op2;
        op2();
    }
};

template<typename>
struct TypeTraits;

#define DEF_TYPE_TRAIT(a) template<> struct TypeTraits<a> { static char* name() {return #a;}};

DEF_TYPE_TRAIT(Add<int>)
DEF_TYPE_TRAIT(Sub<int>)
DEF_TYPE_TRAIT(Mul<int>)
DEF_TYPE_TRAIT(Div<int>)
DEF_TYPE_TRAIT(Mad<int>)
DEF_TYPE_TRAIT(Min<int>)

DEF_TYPE_TRAIT(Add<unsigned int>)
DEF_TYPE_TRAIT(Sub<unsigned int>)
DEF_TYPE_TRAIT(Mul<unsigned int>)
DEF_TYPE_TRAIT(Div<unsigned int>)
DEF_TYPE_TRAIT(Mad<unsigned int>)
DEF_TYPE_TRAIT(Min<unsigned int>)

DEF_TYPE_TRAIT(Add<float>)
DEF_TYPE_TRAIT(Sub<float>)
DEF_TYPE_TRAIT(Mul<float>)
DEF_TYPE_TRAIT(Div<float>)
DEF_TYPE_TRAIT(Mad<float>)
DEF_TYPE_TRAIT(Min<float>)

DEF_TYPE_TRAIT(Add<long long int>)
DEF_TYPE_TRAIT(Sub<long long int>)
DEF_TYPE_TRAIT(Mul<long long int>)
DEF_TYPE_TRAIT(Div<long long int>)
DEF_TYPE_TRAIT(Mad<long long int>)
DEF_TYPE_TRAIT(Min<long long int>)

DEF_TYPE_TRAIT(Add<double>)
DEF_TYPE_TRAIT(Sub<double>)
DEF_TYPE_TRAIT(Mul<double>)
DEF_TYPE_TRAIT(Div<double>)
DEF_TYPE_TRAIT(Mad<double>)
DEF_TYPE_TRAIT(Min<double>)

DEF_TYPE_TRAIT(Shfl<int>)
DEF_TYPE_TRAIT(Ballot<int>)
DEF_TYPE_TRAIT(All<int>)
DEF_TYPE_TRAIT(Any<int>)
DEF_TYPE_TRAIT(Sync)
DEF_TYPE_TRAIT(Prmt)

} // namespace cumib

#endif // __CUMIB_OPERATIONS_CUH__