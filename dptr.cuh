#ifndef __CUMIB_DPTR_CUH__
#define __CUMIB_DPTR_CUH__

#include <builtin_types.h>

template <typename T>
struct DPtr
{
    typedef T             value_type;
    typedef unsigned int  index_type;
    typedef unsigned char inner_ptr_type;

    __host__ __device__ __forceinline__       T* row(index_type y)       { return (      T*)( data + y * step); }
    __host__ __device__ __forceinline__ const T* row(index_type y) const { return (const T*)( data + y * step); }

    __host__ __device__ __forceinline__       T& operator ()(index_type y, index_type x)       { return row(y)[x]; }
    __host__ __device__ __forceinline__ const T& operator ()(index_type y, index_type x) const { return row(y)[x]; }

    __host__ __device__ __forceinline__ DPtr() : data(0), step(0) {}

    template<typename T1>
    __device__ __forceinline__ void operator ()(index_type y, index_type x, const T1& v)
    {
        ((T1*)( data + y * step))[x] = v;
    }

    template<typename M>
    __host__ DPtr(const M& m)
    {
        data = m.data;
        step = m.step;
    }

    inner_ptr_type* data;
    size_t step;
};

typedef unsigned int int32u;
typedef   signed int int32s;

typedef unsigned char int8u;
typedef   signed char int8s;

typedef DPtr<int8u>  DPtrb;
typedef DPtr<int32u> DPtru;

#endif // __CUMIB_DPTR_CUH__
