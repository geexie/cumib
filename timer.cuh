#ifndef __CUMIB_TIMER_CUH__
#define __CUMIB_TIMER_CUH__

#include <string>
#include <stdint.h>
#include <inttypes.h>
#include <iostream>
#include <cmath>
#include <assert.h>

typedef uint64_t uint64;
typedef int64_t   int64;

class timer
{
public:
    timer(const std::string& _name = ""):
    total(0), current(0), n(0), freq(getTickFrequency()), name(_name) {}

    void start() { ++n; current = getTickCount(); }
    void end()   {total += getTickCount() - current;}

    ~timer()
    {
        if (n == 0) return;
        std::cout
        << name << " => "
        << n << " x "
        << (total / freq / n * 1000) << "ms  =  "
        << (total / freq * 1000) << "ms"
        << std::endl;
    }

private:
    uint64 total;
    uint64 current;
    int n;
    double freq;
    std::string name;

    static int64 getTickCount(void)
    {
#if defined __linux || defined __linux__
        struct timespec tp;
        clock_gettime(CLOCK_MONOTONIC, &tp);
        return (int64)tp.tv_sec*1000000000 + tp.tv_nsec;
#else
        struct timeval tv;
        struct timezone tz;
        gettimeofday( &tv, &tz );
        return (int64)tv.tv_sec*1000000 + tv.tv_usec;
#endif
    }

    static double getTickFrequency(void)
    {
#if defined __linux || defined __linux__
        return 1e9;
#else
        return 1e6;
#endif
    }
};

enum {
    TEST_SYCLES = 100
};

#undef TIMERON
#undef TIMEROFF
#undef TIMER

#define TIMER(suffix, str) static timer timer_##suffix(str);

#define TIMERON(suffix) timer_##suffix.start()

#define TIMEROFF(suffix) timer_##suffix.end()

// always assume byte type
template<class M1, class M2>
static int countNonZeros(const M1& mat1, const M2& mat2)
{
    assert (mat1.cols == mat2.cols);
    assert (mat1.rows == mat2.rows);
    assert (mat1.rows == mat2.rows);
    int res =0;
    for(int row = 0; row < mat1.rows; ++row)
        for(int col = 0; col < mat1.cols; ++col)
            res += (std::abs(mat1.row(row)[col] - mat2.row(row)[col]) != 0);
    return res;
}

template<class M1, class M2>
static int countNonZerosApprox(const M1& mat1, const M2& mat2, float eps)
{
    assert (mat1.cols == mat2.cols);
    assert (mat1.rows == mat2.rows);
    assert (mat1.rows == mat2.rows);
    int res =0;
    for(int row = 0; row < mat1.rows; ++row)
        for(int col = 0; col < mat1.cols; ++col)
            res += (std::abs(mat1.row(row)[col] - mat2.row(row)[col]) > eps);
    return res;
}

template<class M1, class M2>
static void assert_binary_equal(const M1& ref, const M2& res, std::string nf, std::string n1, std::string n2)
{
    int diff_pixels = countNonZeros(ref, res);
    if (diff_pixels)
    {
        std::cout << "[\033[1;31m test failed\033[0m ] in " << nf << " " << n1 << " and " << n2
                  << " have diff pixels " << diff_pixels << std::endl;
    }
    else
    {
        std::cout << "[ \033[1;32m test passed\033[0m ] in " << nf << " " << n1 << " and " << n2 << " is equal"
                  << std::endl;
    }
    assert(!diff_pixels);
}

template<class M1, class M2>
static void assert_float_equal(const M1& ref, const M2& res, float eps, std::string nf, std::string n1, std::string n2)
{
    int diff_pixels = countNonZerosApprox(ref, res, eps);

    if (diff_pixels)
    {
        std::cout << "[\033[1;31m test failed\033[0m ] in " << nf << " " << n1 << " and " << n2 << " have diff pixels "
                  << diff_pixels << " greater then " << eps << std::endl;
    }
    else
    {
        std::cout << "[ \033[1;32m test passed\033[0m ] in " << nf << " " << n1 << " and " << n2 << " is near with "
                  << eps << std::endl;
    }
    assert(!diff_pixels);
}

#define ASSERT_BINARY_EQUAL(f, ref, res)     assert_binary_equal(ref, res, #f, #ref, #res)

#define ASSERT_FLOAT_EQUAL(f, ref, res, eps) assert_float_equal(ref, res, eps, #f, #ref, #res)

#define PERF_TEST(timer, f)  do { f;    \
TIMER(timer, #f);                       \
for (int i = 0; i < TEST_SYCLES; ++i)   \
{                                       \
    TIMERON(timer);                     \
    f;                                  \
    TIMEROFF(timer);                    \
}                                       \
}while(0)

#endif // __CUMIB_TIMER_CUH__
