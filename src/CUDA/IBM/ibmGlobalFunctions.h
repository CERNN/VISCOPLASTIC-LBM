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
__device__ __inline__ dfloat atomicDoubleAdd(double* address, double val)
{
    #if __CUDA_ARCH__ < 600
    //TODO add cuda version and double/float swap
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);

    #else
        atomicAdd(address,val);
    #endif
}

/*
*   @brief Evaluate the force distributions based on the stencil 
*   @param x: the distance between the node thand the reference position
*   @return force weight
*/
__device__ __forceinline__  dfloat stencil(dfloat x) {
    dfloat x = abs(x);
    #if defined STENCIL_2
        if (x > 1.0) {
            return 0.0;
        }
        else {
            return (1 - x);
        }
    #elif defined STENCIL_4
        if (x <= 1) {
            return (1.0 / 8.0)*(3.0 - 2.0*x + sqrt(1 + 4 * x - 4 * x*x));;
        }
        else if (x > 1.0 && x <= 2.0) {
            return (1.0 / 8.0)*(5.0 - 2.0*x - sqrt(-7.0 + 12.0*x - 4.0*x*x));
        }
        else {
            return 0.0;
        }
    #endif //STENCIL
}

#endif // !__IBM_IDX_H