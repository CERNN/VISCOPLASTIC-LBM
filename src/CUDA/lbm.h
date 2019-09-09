/*
*   @file lbm.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief LBM steps: boundary conditions, collision, macroscopics, stream 
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __LBM_H
#define __LBM_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "structs\macroscopics.h"
#include "structs\macrProc.h"
#include "boundaryConditionsHandler.h"


/*
*   @brief Applies boundary conditions, updates macroscopics and then performs
*          collision and streaming
*   @param pop: populations and boundary conditions to use
*   @param macr: macroscopics to use/update
*   @param save: save macroscopics
*   @param step: simulation step
*/
__global__
void gpuBCMacrCollisionStream(
    dfloat* const pop,
    dfloat* const popAux,
    NodeTypeMap* const mapBC,
    Macroscopics* const macr,
    bool const save,
    int const step
);


/*
*   @brief Update macroscopics of all nodes
*   @param pop: populations to use
*   @param macr: macroscopics to update
*/
__global__
void gpuUpdateMacr(
    Populations* pop,
    Macroscopics* macr
);


#endif // __LBM_H
