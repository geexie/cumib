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

template<typename T>
struct And
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        T tmp;
        asm volatile ("add.s32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp;
    }
};

template<typename T>
struct Sub
{
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a-b; }
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


} // namespace cumib

#endif // __CUMIB_OPERATIONS_CUH__