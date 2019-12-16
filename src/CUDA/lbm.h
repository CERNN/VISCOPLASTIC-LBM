/*
*   @file lbm.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief LBM steps: boundary conditions, collision, macroscopics, stream 
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __LBM_H
#define __LBM_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "structs\macroscopics.h"
#include "structs\macrProc.h"
#include "boundaryConditionsHandler.h"


/*
*   @brief Updates macroscopics and then performs collision and streaming
*   @param pop: populations to use
*   @param popAux: auxiliary populations to stream to
*   @param mapBC: boundary conditions map
*   @param macr: macroscopics to use/update
*   @param save: save macroscopics
*   @param step: simulation step
*/
__global__
void gpuMacrCollisionStream(
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


/*
*   @brief Applies boundary conditions
*   @param mapBC: boundary conditions map
*   @param popPostStream: populations post streaming to update
*   @param popPostCol: populations post collision to use
*   @param idxsBCNodes: vector of scalar indexes of boundary conditions
*   @param totalBCNodes: total number of nodes boundary conditions
*/
__global__
void gpuApplyBC(NodeTypeMap* mapBC, 
    dfloat* popPostStream,
    dfloat* popPostCol,
    size_t* idxsBCNodes,
    size_t totalBCNodes
);


#endif // __LBM_H
