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

#ifndef __CUMIB_CUMIB_CUH__
#define __CUMIB_CUMIB_CUH__

#if _WIN32 || _WIN64
# if _WIN64
#  define __CUMIB_ENV_64__
# else
#  define __CUMIB_ENV_32__
# endif
#endif

#if __GNUC__
# if __x86_64__ || __ppc64__
#  define __CUMIB_ENV_64__
# else
#  define __CUMIB_ENV_32__
# endif
#endif

#if defined(__CUMIB_ENV_64__)
# define __CUMIB_PTR_REG__ "l"
#else
# define __CUMIB_PTR_REG__ "r"
#endif

namespace cumib {

class CudaTimer {

public:
    CudaTimer()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    ~CudaTimer()
    {
        cudaEventDestroy( _start );
        cudaEventDestroy( _stop );
    }

    void go()
    {
        cudaEventRecord( _start, 0 );
    }

    //returns time after _start event to current point in Ms.
    float measure()
    {
        cudaEventRecord( _stop, 0 );
        cudaEventSynchronize( _stop );
        float time;
        cudaEventElapsedTime( &time, _start, _stop );
        return time;
    }

private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
};

}

#endif // __CUMIB_CUMIB_CUH__