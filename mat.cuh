#ifndef __CUMIB_MAT_CUH__
#define __CUMIB_MAT_CUH__

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cstdlib>
#include <limits>
#include <iostream>

#include "cuda_assert.cuh"

// simple Mat implementation
template<typename Ptr> struct AllocGmemPitched : Ptr
{
    void allocate(int rows, int cols)
    {
        size_t pitch = 0;
        void* ptr = 0;
        cuda_assert(cudaMallocPitch(&ptr, &pitch, cols * sizeof(typename Ptr::value_type), rows));

        Ptr::step = pitch;
        Ptr::data = (typename Ptr::inner_ptr_type*)ptr;
    }

    void deallocate() { cuda_assert(cudaFree(Ptr::data)); }
};

template<typename Ptr> struct AllocNew : Ptr
{
    void allocate(int rows, int cols)
    {
        int pitch = ((cols * sizeof(typename Ptr::value_type) + 64 - 1) / 64) * 64;
        void* ptr = malloc(rows * pitch);

        Ptr::step = pitch;
        Ptr::data = (typename Ptr::inner_ptr_type*)ptr;
    }

    void deallocate() { free(Ptr::data); }
};

template<typename T, template <class S> class AllocPolicy>
struct DMat : AllocPolicy<DPtr<T> >
{
    DMat(int _rows, int _cols) : rows(_rows), cols(_cols)
    {
        AllocPolicy<DPtr<T> >::allocate(rows, cols);
    }

    ~DMat() {AllocPolicy<DPtr<T> >::deallocate();}

    int elemSize() const {return sizeof(T);}

    int rows;
    int cols;
};

typedef DMat<int8u, AllocNew>          HMatb;
typedef DMat<int8u, AllocGmemPitched>  DMatb;

typedef DMat<int32u, AllocNew>          HMatu;
typedef DMat<int32u, AllocGmemPitched>  DMatu;

template<typename T, template <class S> class P1, template <class S> class P2>
static void copy(DMat<T, P1>& src, DMat<T, P2>& dst);

template<>
void copy<int8u, AllocNew, AllocGmemPitched>(DMat<int8u, AllocNew>& src, DMat<int8u, AllocGmemPitched>& dst)
{
    cuda_assert(cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice));
    cuda_assert(cudaGetLastError());
}

template<>
void copy<int8u, AllocGmemPitched, AllocNew>(DMat<int8u, AllocGmemPitched>& src, DMat<int8u, AllocNew>& dst)
{
    cuda_assert(cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost));
    cuda_assert(cudaGetLastError());
}

template<>
void copy<int32u, AllocNew, AllocGmemPitched>(DMat<int32u, AllocNew>& src, DMat<int32u, AllocGmemPitched>& dst)
{
    cuda_assert(cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice));
    cuda_assert(cudaGetLastError());
}

template<>
void copy<int32u, AllocGmemPitched, AllocNew>(DMat<int32u, AllocGmemPitched>& src, DMat<int32u, AllocNew>& dst)
{
    cuda_assert(cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost));
    cuda_assert(cudaGetLastError());
}

template<class M>
static void fill(M& m)
{
    for(int row = 0; row < m.rows; ++row)
        for(int col = 0; col < m.cols; ++col)
            m.row(row)[col] = (row * col) % std::numeric_limits<unsigned char>::max();
}

namespace std {

std::ostream& operator<< (std::ostream& stream, const HMatb& m)
{
    for(int row = 0; row < m.rows; ++row)
    {
        for(int col = 0; col < m.cols; ++col)
            stream << (int)m.row(row)[col] << ", " ;
        stream << std::endl;
    }
    return stream;
}

std::ostream& operator<< (std::ostream& stream, const HMatu& m)
{
    for(int row = 0; row < m.rows; ++row)
    {
        for(int col = 0; col < m.cols; ++col)
            stream << (int)m.row(row)[col] << ", " ;
        stream << std::endl;
    }
    return stream;
}

} // namespace std {

#endif // __CUMIB_MAT_CUH__