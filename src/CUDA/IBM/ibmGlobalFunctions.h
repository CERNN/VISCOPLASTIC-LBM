/*
*   @file globalFunctions.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief index functions utilized in the immersed boundary method
*   @version 0.0.1
*   @date 26/08/2020
*/

#ifndef __IBM_GLOBAL_FUNCTIONS_H
#define __IBM_GLOBAL_FUNCTIONS_H


#include "ibmVar.h"


// Only compile for compute capability lower than 6.0, since for 6.0 or higher
// this functionality already exists
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  #if __CUDA_ARCH__ >= 600
/*
*   @brief Double atomic add for Cuda capabailities less than 6.0
*   @param address: memory address where will be added the value
*   @param value: added value
*   @example: 
*           __global__ void addKernel(double *a)
*           {
*               double addedValue = 1.0;
*               atomicDoubleAdd(a, addedValue);
*           }
*/
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
  }

#endif

/*
*   @brief Evaluate the force distributions based on the stencil
*
*   @param x: the distance between the node thand the reference position
*   @return force weight
*/
__device__ __forceinline__  dfloat stencil(dfloat x) {
    dfloat absX = abs(x);
    #if defined STENCIL_2
        if (absX > 1.0) {
            return 0.0;
        }
        else {
            return (1 - x);
        }
    #elif defined STENCIL_4
        if (absX <= 1) {
            return (1.0 / 8.0)*(3.0 - 2.0*absX + sqrt(1 + 4 * absX - 4 * absX*absX));
        }
        else if (absX > 1.0 && absX <= 2.0) {
            return (1.0 / 8.0)*(5.0 - 2.0*absX - sqrt(-7.0 + 12.0*absX - 4.0*absX*absX));
        }
        else {
            return 0.0;
        }
    #endif //STENCIL
}

#endif // !__IBM_GLOBAL_FUNCTIONS_H
