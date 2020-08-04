/*
*   @file populations.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for populations and boundary conditions map 
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __POPULATIONS_H
#define __POPULATIONS_H

#include "../var.h"
#include "errorDef.h"
#include "nodeTypeMap.h"
#include <cuda.h>

/*
*   Struct for LBM populations and the node's boundary conditions
*/
typedef struct populations
{
public:    
    dfloat* pop;            // Populations
    dfloat* popAux;         // Auxiliary populations
    NodeTypeMap* mapBC;     // Boundary conditions map

    /* Constructor */
    __host__
    populations()
    {
        this->pop = nullptr;
        this->popAux = nullptr;
        this->mapBC = nullptr;
    }

    /* Destructor */
    __host__
    ~populations()
    {
        this->pop = nullptr;
        this->popAux = nullptr;
        this->mapBC = nullptr;
    }

    /* Allocate populations */
    __host__
    void popAllocation()
    {
        checkCudaErrors(cudaMallocManaged((void**)&(this->pop), memSizePop));
        checkCudaErrors(cudaMallocManaged((void**)&(this->popAux), memSizePop));
        checkCudaErrors(cudaMalloc((void**)&(this->mapBC), memSizeMapBC));
    }

    /* Free populations */
    __host__
    void popFree()
    {
        checkCudaErrors(cudaFree(this->pop));
        checkCudaErrors(cudaFree(this->popAux));
        checkCudaErrors(cudaFree(this->mapBC));
    }

    /* Swap populations pointers */
    __host__ __device__
    void __forceinline__ swapPop()
    {
        dfloat* tmp = pop;
        pop = popAux;
        popAux = tmp;
    }
} Populations;


#endif // __POPULATIONS_H