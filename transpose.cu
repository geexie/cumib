#include "cudassert.cuh"
#include "perf_test.cuh"
#include "mat.cuh"
#include <assert.h>

__device__ __forceinline__ unsigned int lane()
{
#ifdef __CUDA_ARCH__
    unsigned int ret; asm("mov.u32 %0, %laneid;" : "=r"(ret) ); return ret;
#else
    return 0;
#endif
}

__host__  __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

enum
{
    TRANSPOSE_TILE_DIM_X       = 32,
    TRANSPOSE_TILE_DIM_Y       = 64,
    TRANSPOSE_TILE_DIM         = TRANSPOSE_TILE_DIM_X,
    TRANSPOSE_BLOCK_ROWS       = 8,
    TRANSPOSE_TILE_DIM_BK_FREE = TRANSPOSE_TILE_DIM + 1,

    SHUFFLE_ELEMENTS_PERF_WARP      = 8,
    SHUFFLE_ELEMENTS_PERF_WARP_HALF = SHUFFLE_ELEMENTS_PERF_WARP >> 1,
    SHUFFLE_ELEMENTS_VECTORS        = SHUFFLE_ELEMENTS_PERF_WARP / 4,
    SHUFFLE_TRANSPOSE_BLOCK_X       = 32,
    SHUFFLE_TRANSPOSE_BLOCK_Y       = 4,
    SHUFFLE_TRANSPOSE_BLOCK_ROWS    = SHUFFLE_TRANSPOSE_BLOCK_Y * SHUFFLE_ELEMENTS_PERF_WARP
};

template <typename T>
__global__ void copy(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(yIndex + i)[xIndex] = idata.row(yIndex + i)[xIndex];
    }
}

void runCopy(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    copy<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template<typename T>
__global__ void copySharedMem(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM/* + 1*/];

    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata.row(yIndex + i)[xIndex];
    }

    __syncthreads();

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(yIndex + i)[xIndex] = tile[threadIdx.y + i][threadIdx.x];
    }
}

void runCopySharedMem(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    copySharedMem<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template<typename T>
__global__ void copySharedMemPlus1(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata.row(yIndex + i)[xIndex];
    }

    __syncthreads();

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(yIndex + i)[xIndex] = tile[threadIdx.y + i][threadIdx.x];
    }
}

void runCopySharedMemPlus1(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    copySharedMemPlus1<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template <typename T>
__global__ void transposeNaive(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    odata.row(xIndex)[yIndex] = idata.row(yIndex)[xIndex];
}

void runTransposeNaive(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_BLOCK_ROWS));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    transposeNaive<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template <typename T>
__global__ void transposeNaiveBlock(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(xIndex)[yIndex+i] = idata.row(yIndex + i)[xIndex];
    }
}

void runTransposeNaiveBlock(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    transposeNaiveBlock<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template <typename T>
__global__ void transposeCoalesced(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];

    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata.row(yIndex + i)[xIndex];
    }

    xIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    __syncthreads();

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(yIndex + i)[xIndex] = tile[threadIdx.x][threadIdx.y + i];
    }
}

void runTransposeCoalesced(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    transposeCoalesced<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template <typename T>
__global__ void transposeCoalescedPlus1(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata.row(yIndex + i)[xIndex];
    }

    xIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    __syncthreads();

    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
    {
        odata.row(yIndex + i)[xIndex] = tile[threadIdx.x][threadIdx.y + i];
    }
}

void runTransposeCoalescedPlus1(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 grid(divUp(cols, TRANSPOSE_TILE_DIM), divUp(rows, TRANSPOSE_TILE_DIM));
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);

    transposeCoalescedPlus1<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template<typename T> __device__ __forceinline__ T laneIsEven();
template<> __device__ __forceinline__ unsigned int laneIsEven<unsigned int>()
{
    unsigned int laneMask;
    asm("mov.u32 %0, %lanemask_eq;" : "=r"(laneMask));
    return laneMask & 0x55555555;
}

template<typename T>
__global__ void transposeShuffle(const DPtr<T> idata, DPtr<T> odata, int cols, int rows)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int yIndex1 = yIndex * SHUFFLE_ELEMENTS_VECTORS;

    yIndex *= SHUFFLE_ELEMENTS_PERF_WARP;

    int4 reg0, reg1;

    reg0.x = idata.row(yIndex + 0)[xIndex];
    reg0.y = idata.row(yIndex + 1)[xIndex];
    reg0.z = idata.row(yIndex + 2)[xIndex];
    reg0.w = idata.row(yIndex + 3)[xIndex];

    reg1.x = idata.row(yIndex + 4)[xIndex];
    reg1.y = idata.row(yIndex + 5)[xIndex];
    reg1.z = idata.row(yIndex + 6)[xIndex];
    reg1.w = idata.row(yIndex + 7)[xIndex];

    unsigned int isEven = laneIsEven<unsigned int>();
    int4 target = isEven ? reg1 : reg0;

    target.x = __shfl_xor(target.x, 1);
    target.y = __shfl_xor(target.y, 1);
    target.z = __shfl_xor(target.z, 1);
    target.w = __shfl_xor(target.w, 1);

    const int oIndexY = blockIdx.x * blockDim.x + (threadIdx.x >> 1) * 2;
    const int oIndexX = yIndex1 + (isEven == 0);

    if (isEven) reg1 = target; else reg0 = target;

    odata(oIndexY + 0, oIndexX, reg0);
    odata(oIndexY + 1, oIndexX, reg1);
}

void runTransposeShuffle(const DPtru& src, DPtru& dst, int cols, int rows)
{
    dim3 block(SHUFFLE_TRANSPOSE_BLOCK_X, SHUFFLE_TRANSPOSE_BLOCK_Y);
    dim3 grid(divUp(cols, SHUFFLE_TRANSPOSE_BLOCK_X), divUp(rows, SHUFFLE_TRANSPOSE_BLOCK_ROWS));

    transposeShuffle<<<grid, block>>>(src, dst, cols, rows);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template<class M1, class M2>
void referencetranspose(const M1& src, M2& dst)
{
    for(int row = 0; row < src.rows; ++row)
        for(int col = 0; col < src.cols; ++col)
            dst.row(col)[row] = src.row(row)[col];
}

typedef void (*transpose_t) (const DPtru&, DPtru&, int32s, int32s);

void testTranspoce(int ih, int iw,  transpose_t func)
{
    HMatu  src(ih, iw),  dst(iw, ih), res(iw, ih);
    DMatu dsrc(ih, iw), ddst(iw, ih);
    assert(src.cols == dst.rows);
    assert(src.rows == dst.cols);

    fill(src);
    copy(src, dsrc);

    referencetranspose(src, dst);

    func(dsrc, ddst, src.cols, src.rows);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(func, dst, res);
}

template<class M1, class M2, class M3>
void benchmark(const M1& src, M2& dst, M3& dst1)
{
    PERF_TEST(t_copy1, runCopy(src, dst1, src.cols, src.rows));
    PERF_TEST(t_copy2, runCopySharedMem(src, dst1, src.cols, src.rows));
    PERF_TEST(t_copy2, runCopySharedMemPlus1(src, dst1, src.cols, src.rows));

    PERF_TEST(t_npp1, runTransposeNaive(src, dst, src.cols, src.rows));
    PERF_TEST(t_npp2, runTransposeNaiveBlock(src, dst, src.cols, src.rows));
    PERF_TEST(t_npp3, runTransposeCoalesced(src, dst, src.cols, src.rows));
    PERF_TEST(t_npp3, runTransposeCoalescedPlus1(src, dst, src.cols, src.rows));
    PERF_TEST(t_npp4, runTransposeShuffle(src, dst, src.cols, src.rows));
}

int main(int argc, char** argv)
{
    int deviceId = 0;
    if (argc > 1)
    {
        deviceId = atoi(argv[1]);
    }

    cuda_assert(cudaSetDevice(deviceId));
    printCudaDeviceInfo(deviceId);

    const int32s iw = 1920;
    const int32s ih = 1080;

    testTranspoce(ih, iw, runTransposeNaive);
    testTranspoce(ih, iw, runTransposeNaiveBlock);
    testTranspoce(ih, iw, runTransposeCoalesced);
    testTranspoce(ih, iw, runTransposeCoalescedPlus1);
    testTranspoce(ih, iw, runTransposeShuffle);

    DMatu dsrc(ih, iw), ddst(iw, ih), ddst1(ih, iw);
    benchmark(dsrc, ddst, ddst1);

    return 0;
}
