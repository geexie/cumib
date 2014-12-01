#ifndef __CUMIB_CUDA_ACCERT_CUH__
#define __CUMIB_CUDA_ACCERT_CUH__

#include <stdio.h>
#include <builtin_types.h>

static void __cuda_assert(cudaError_t err, const char* file, const int line, const char* func)
{
    if (err != cudaSuccess)
    {
        printf("Error : %s in %s, file %s, line %d \n", cudaGetErrorString(err), func, file, line);
        fflush(stdout); exit(-1);
    }
}


#define cuda_assert(expr) __cuda_assert((expr), __FILE__, __LINE__, __func__)

#endif // __CUMIB_CUDA_ACCERT_CUH__