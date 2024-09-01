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
