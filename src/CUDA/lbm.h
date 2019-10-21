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
*   @brief Applies local boundary conditions, updates macroscopics and then 
*          performs collision and streaming
*   @param pop: populations to use
*   @param popAux: auxiliary populations to stream to
*   @param mapBC: boundary conditions map
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


/*
*   @brief Applies non local boundary conditions
*   @param pop: populations to use
*   @param mapBC: boundary conditions map
*   @param popAuxNonLocal: auxiliary populations to keep the boundary conditions result
*   @param idxNonLocal: scalar index of non local boundary conditions
*   @param totalNonLocalBC: total number of nodes with non local boundary conditions
*/
__global__
void gpuApplyNonLocalBC(dfloat* pop, 
    NodeTypeMap* mapBC, 
    dfloat* popAuxNonLocal, 
    size_t* idxNonLocal,
    size_t totalNonLocalBC
);


/*
*   @brief Synchronize non local boundary conditions applied
*   @param pop: populations to copy to
*   @param popAuxNonLocal: auxiliary populations to copy from
*   @param idxNonLocal: scalar index of non local boundary conditions
*   @param totalNonLocalBC: total number of nodes with non local boundary conditions
*/
__global__
void gpuSynchronizeNonLocalBC(dfloat* pop,
    dfloat* popAuxNonLocal,
    size_t* idxNonLocal,
    size_t totalNonLocalBC
);

#endif // __LBM_H
