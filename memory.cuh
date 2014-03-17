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

#ifndef __CUMIB_MEMORY_CUH__
#define __CUMIB_MEMORY_CUH__

#include "cumib.cuh"

namespace cumib {

template<typename T> __device__ __forceinline__ T ldCa(const T* ptr);
template<typename T> __device__ __forceinline__ T ldCg(const T* ptr);
template<typename T> __device__ __forceinline__ T ldCs(const T* ptr);
template<typename T> __device__ __forceinline__ T ldCu(const T* ptr);
template<typename T> __device__ __forceinline__ T ldCv(const T* ptr);

template<> __device__ __forceinline__ int ldCa<int>(const int* ptr)
{
    int val; asm("ld.global.ca.s32 %0, [%1];" : "=r"(val) : __CUMIB_PTR_REG__(ptr)); return val;
}

template<> __device__ __forceinline__ int ldCg<int>(const int* ptr)
{
    int val; asm("ld.global.cg.s32 %0, [%1];" : "=r"(val) : __CUMIB_PTR_REG__(ptr)); return val;
}

template<> __device__ __forceinline__ int ldCs<int>(const int* ptr)
{
    int val; asm("ld.global.cs.s32 %0, [%1];" : "=r"(val) : __CUMIB_PTR_REG__(ptr)); return val;
}

template<> __device__ __forceinline__ int ldCu<int>(const int* ptr)
{
    int val; asm("ld.global.lu.s32 %0, [%1];" : "=r"(val) : __CUMIB_PTR_REG__(ptr)); return val;
}

template<> __device__ __forceinline__ int ldCv<int>(const int* ptr)
{
    int val; asm("ld.global.cv.s32 %0, [%1];" : "=r"(val) : __CUMIB_PTR_REG__(ptr)); return val;
}

template<typename T>
struct Nop
{
    __device__ __forceinline__ T* operator()(T *j) { asm volatile ("" : : : "memory"); return j; }
};

template<typename T>
struct Ld
{
    __device__ __forceinline__ T* operator()(const T *j) { return *(T **)j;}
};

template<typename T> struct Ldg;

#if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
# if __CUMIB_ENV_64__

template<typename T>
struct Ldg
{
    __device__ __forceinline__ T* operator()(const T* ptr)
    {
        unsigned long long ret;
        asm volatile ("ld.global.nc.u64 %0, [%1];"  : "=l"(ret) : __CUMIB_PTR_REG__(ptr));
        return reinterpret_cast<T*>((T)ret);
    }
};

# else
template<typename T>
struct Ldg
{
    __device__ __forceinline__ T* operator()(const T* ptr)
    {
        unsigned int ret;
        asm volatile ("ld.global.nc.u32 %0, [%1];"  : "=r"(ret) : __CUMIB_PTR_REG__(ptr));
        return reinterpret_cast<T*>((T)ret);
    }
};
# endif
#else
// stub, since non supported
template<typename T>
struct Ldg
{
    __device__ __forceinline__ T* operator()(const T *j) { return *(T **)j;}
};
#endif

template<typename T>
struct TestPrint
{
    T* operator()(T* ptr) const { printf("%p\n", ptr); ++ptr; return ptr;}
};

} // namespace cumib {

#endif // __CUMIB_MEMORY_CUH__