#include "cuda_assert.cuh"
#include "timer.cuh"
#include "dptr.cuh"
#include "mat.cuh"

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

enum {
    WARP_SHIFT = 5,
    TRESHOLD_UNROLL_FACTOR = 2
};

__global__ void threshold_bw(const DPtrb src, DPtrb dst, int32s cols, int32s rows, int8u threshold)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cols && y < rows)
        dst.row(y)[x] = max(threshold, src.row(y)[x]);
}

void runThresholdBW(const DPtrb& src, DPtrb& dst, int32s cols, int32s rows, int8u threshold,
    int32s blockSize_x, int32s blockSize_y)
{
    threshold_bw<<<dim3(divUp(cols,blockSize_x), divUp(rows, blockSize_y)), dim3(blockSize_x, blockSize_y)>>>(
        src, dst, cols, rows, threshold);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

__global__ void threshold_ww_row(const DPtrb src, DPtrb dst, int32s cols, int32s rows, int8u threshold)
{
    const int bid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x >> WARP_SHIFT);
    const int y = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize + bid;

    if (y < rows)
        for (unsigned int x = lane(); x < cols; x += warpSize)
            dst.row(y)[x] = max(threshold, src.row(y)[x]);
}

void runThresholdWWRow(const DPtrb& src, DPtrb& dst, int32s cols, int32s rows, int8u threshold, int32s blockSize)
{
    threshold_ww_row<<<dim3(divUp(rows, blockSize >> WARP_SHIFT)), dim3(blockSize)>>>(src, dst, cols, rows, threshold);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

__global__ void threshold_ww_row_unr(const DPtrb src, DPtrb dst, int32s cols, int32s rows, int8u threshold)
{
    const int block_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x >> WARP_SHIFT);
    int y = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize + block_id;
    y *= TRESHOLD_UNROLL_FACTOR;

    int8u tmp[TRESHOLD_UNROLL_FACTOR];

    if (y + TRESHOLD_UNROLL_FACTOR < rows)
    {
        for (int x = lane(); x < cols; x += warpSize)
        {
            // fix me: add unrolled loop
            tmp[0] = max(threshold, src.row(y + 0)[x]);
            tmp[1] = max(threshold, src.row(y + 1)[x]);

            dst.row(y + 0)[x] = tmp[0];
            dst.row(y + 1)[x] = tmp[1];
        }
    }
    else
    {
        for (;y < rows; ++y)
        {
             int x = lane();
            for (; x < cols; x += warpSize)
            {
                dst.row(y)[x] = max(threshold, src.row(y)[x]);
            }
        }
    }
}

void runThresholdWWRowU2(const DPtrb& src, DPtrb& dst, int32s cols, int32s rows, int8u threshold, int32s blockSize)
{
    threshold_ww_row_unr<<<dim3(divUp(rows, (blockSize >> WARP_SHIFT) * TRESHOLD_UNROLL_FACTOR)), dim3(blockSize)>>>(
        src, dst, cols, rows, threshold);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

static unsigned int uchar_mask(unsigned char x) { return x * 0x01010101; }

template<typename T> __device__ __forceinline__ T vmin_u8(T,T);

template<> __device__ __forceinline__ unsigned int vmin_u8(unsigned int v, unsigned int m)
{
    unsigned int res = 0;
    asm("vmax4.u32.u32.u32 %0.b3210, %1.b3210, %2.b7654, %3;" : "=r"(res) : "r"(v), "r"(m), "r"(0));
    return res;
}

__global__ void threshold_ww_row_x4(const DPtru src, DPtru dst, int32s cols, int32s rows, int32u mask)
{
    const int bid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x >> WARP_SHIFT);
    const int y = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize + bid;

    if (y < rows)
        for (int x = lane(); x < cols / sizeof(int8u); x += warpSize)
        {
            int32u tmp = src.row(y)[x];
            int32u res = vmin_u8(tmp, mask);
            dst.row(y)[x] = res;
        }
}

void runThresholdWWRowX4(const DPtru& src, DPtru& dst, int32s cols, int32s rows, int8u threshold, int32s blockSize)
{
    unsigned int m = uchar_mask(threshold);
    threshold_ww_row_x4<<<dim3(divUp(rows, blockSize >> 5)), dim3(blockSize)>>>(src, dst, cols, rows, m);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

__global__ void threshold_ww_row_unr_x4(const DPtru src, DPtru dst, int32s cols, int32s rows, int32u mask)
{
    const int block_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x >> WARP_SHIFT);
    int y = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize + block_id;
    y *= 2;

    unsigned int tmp0;
    unsigned int tmp1;

    unsigned int res0;
    unsigned int res1;

    if (y + 2 < rows)
    {
        for (int x = lane(); x < cols / sizeof(int32u); x += warpSize)
        {
            tmp0 = src.row(y)[x];
            tmp1 = src.row(y + 1)[x];

            res0 = vmin_u8(tmp0, mask);
            res1 = vmin_u8(tmp1, mask);

            dst.row(y)[x] = res0;
            dst.row(y + 1)[x] = res1;
        }
    }
    else
    {
        for (;y < rows; ++y)
        {
             int x = lane();
            for (; x < cols / sizeof(unsigned int); x += warpSize)
            {
                dst.row(y)[x] = vmin_u8(src.row(y)[x], mask);
            }
        }
    }
}

void runThresholdWWRowU2X4(const DPtru& src, DPtru&dst, int32s cols, int32s rows, int8u threshold, int32s blockSize)
{
    unsigned int m = uchar_mask(threshold);
    threshold_ww_row_unr_x4<<<dim3(divUp(rows, (blockSize >> WARP_SHIFT) * TRESHOLD_UNROLL_FACTOR)), dim3(blockSize)>>>(
        src, dst, cols, rows, m);

    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
}

template<class M1, class M2, typename T>
void referenceThreshold(const M1& src, M2& dst, T th)
{
    for(int row = 0; row < src.rows; ++row)
        for(int col = 0; col < src.cols; ++col)
            dst.row(row)[col] = std::max(src.row(row)[col], th);
}

void testRunThresholdBW(int ih, int iw, int8u th)
{
    HMatb  src(ih, iw),  dst(ih, iw), res(ih, iw);
    DMatb dsrc(ih, iw), ddst(ih, iw);
    fill(src);
    copy(src, dsrc);

    referenceThreshold(src, dst, th);

    runThresholdBW(dsrc, ddst, src.cols, src.rows, th, 32, 8);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(runThresholdBW, dst, res);
}

void testRunThresholdWWRow(int ih, int iw, int8u th)
{
    HMatb  src(ih, iw),  dst(ih, iw), res(ih, iw);
    DMatb dsrc(ih, iw), ddst(ih, iw);
    fill(src);
    copy(src, dsrc);

    referenceThreshold(src, dst, th);

    runThresholdWWRow(dsrc, ddst, src.cols, src.rows, th, 128);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(runThresholdWWRow, dst, res);
}

void testRunThresholdWWRowU2(int ih, int iw, int8u th)
{
    HMatb  src(ih, iw),  dst(ih, iw), res(ih, iw);
    DMatb dsrc(ih, iw), ddst(ih, iw);
    fill(src);
    copy(src, dsrc);

    referenceThreshold(src, dst, th);

    runThresholdWWRowU2(dsrc, ddst, src.cols, src.rows, th, 128);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(runThresholdWWRowU2, dst, res);
}

void testRunThresholdWWRowX4(int ih, int iw, int8u th)
{
    HMatb  src(ih, iw),  dst(ih, iw), res(ih, iw);
    DMatb dsrc(ih, iw), ddst(ih, iw);
    fill(src);
    copy(src, dsrc);

    referenceThreshold(src, dst, th);

    DPtru ddstu(ddst);
    runThresholdWWRowU2X4(dsrc, ddstu, src.cols, src.rows, th, 128);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(runThresholdWWRowU2X4, dst, res);
}

void testRunThresholdWWRowU2X4(int ih, int iw, int8u th)
{
    HMatb  src(ih, iw),  dst(ih, iw), res(ih, iw);
    DMatb dsrc(ih, iw), ddst(ih, iw);
    fill(src);
    copy(src, dsrc);

    referenceThreshold(src, dst, th);

    DPtru ddstu(ddst);
    runThresholdWWRowU2X4(dsrc, ddstu, src.cols, src.rows, th, 128);
    copy(ddst, res);

    ASSERT_BINARY_EQUAL(runThresholdWWRowU2X4, dst, res);
}

template<class M1, class M2, typename T>
void benchmark(const M1& src, M2& dst, T th)
{
    PERF_TEST(threshold_bw512, runThresholdBW(src, dst, src.cols, src.rows, th, 32, 16));
    PERF_TEST(threshold_bw256, runThresholdBW(src, dst, src.cols, src.rows, th, 32, 8));
    PERF_TEST(threshold_bw160, runThresholdBW(src, dst, src.cols, src.rows, th, 32, 6));
    PERF_TEST(threshold_bw128, runThresholdBW(src, dst, src.cols, src.rows, th, 32, 4));
    PERF_TEST(threshold_bw64,  runThresholdBW(src, dst, src.cols, src.rows, th, 32, 2));
    PERF_TEST(threshold_bw32,  runThresholdBW(src, dst, src.cols, src.rows, th, 32, 1));

    PERF_TEST(threshold_ww16, runThresholdWWRow(src, dst, src.cols, src.rows, th, 512));
    PERF_TEST(threshold_ww8,  runThresholdWWRow(src, dst, src.cols, src.rows, th, 256));
    PERF_TEST(threshold_ww4,  runThresholdWWRow(src, dst, src.cols, src.rows, th, 128));
    PERF_TEST(threshold_ww2,  runThresholdWWRow(src, dst, src.cols, src.rows, th, 64));

    PERF_TEST(threshold_wwu16, runThresholdWWRowU2(src, dst, src.cols, src.rows, th, 512));
    PERF_TEST(threshold_wwu8,  runThresholdWWRowU2(src, dst, src.cols, src.rows, th, 256));
    PERF_TEST(threshold_wwu4,  runThresholdWWRowU2(src, dst, src.cols, src.rows, th, 128));
    PERF_TEST(threshold_wwu2,  runThresholdWWRowU2(src, dst, src.cols, src.rows, th, 64));

    DPtru dstu(dst);
    PERF_TEST(threshold_ww1v, runThresholdWWRowX4(src, dstu, src.cols, src.rows, th, 128));
    PERF_TEST(threshold_ww2v, runThresholdWWRowX4(src, dstu, src.cols, src.rows, th, 64));

    PERF_TEST(threshold_ww1v2, runThresholdWWRowU2X4(src, dstu, src.cols, src.rows, th, 128));
    PERF_TEST(threshold_ww2v2, runThresholdWWRowU2X4(src, dstu, src.cols, src.rows, th, 64));
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        cuda_assert(cudaSetDevice(atoi(argv[1])));
    }

    const int32s iw = 1920;
    const int32s ih = 1080;
    const int8u  th = 127;

    testRunThresholdBW(ih, iw, th);
    testRunThresholdWWRow(ih, iw, th);
    testRunThresholdWWRowU2(ih, iw, th);
    testRunThresholdWWRowX4(ih, iw, th);
    testRunThresholdWWRowU2X4(ih, iw, th);

    DMatb dsrc(ih, iw), ddst(ih, iw);
    benchmark(dsrc, ddst, th);

    return 0;
}