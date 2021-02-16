/*
*   @file gPopulations.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @brief Struct for secondary populations 
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __G_POPULATIONS_H
#define __G_POPULATIONS_H

#include "../var.h"
#include "gVar.h"
#include "../errorDef.h"
#include <cuda.h>

/*
*   Struct for LBM secondary populations and the node's boundary conditions
*/
typedef struct populations
{
public:    
    dfloat* gPop;            // Populations
    dfloat* gPopAux;         // Auxiliary populations
    //NodeTypeMap* gMapBC;     // Boundary conditions map

    /* Constructor */
    __host__
    gPopulations()
    {
        this->gPop = nullptr;
        this->gPopAux = nullptr;
    //    this->gMapBC = nullptr;
    }

    /* Destructor */
    __host__
    ~gPopulations()
    {
        this->gPop = nullptr;
        this->gPopAux = nullptr;
    //    this->gMapBC = nullptr;
    }

    /* Allocate populations */
    __host__
    void gPopAllocation()
    {
        checkCudaErrors(cudaMallocManaged((void**)&(this->gPop), memSizeGPop));
        checkCudaErrors(cudaMallocManaged((void**)&(this->gPopAux), memSizeGPop));
    //    checkCudaErrors(cudaMalloc((void**)&(this->gMapBC), MEM_SIZE_MAP_BC));
    }

    /* Free populations */
    __host__
    void gPopFree()
    {
        checkCudaErrors(cudaFree(this->gPop));
        checkCudaErrors(cudaFree(this->gPopAux));
    //    checkCudaErrors(cudaFree(this->gMapBC));
    }

    /* Swap populations pointers */
    __host__ __device__
    void __forceinline__ swapPop()
    {
        dfloat* tmp = gPop;
        gPop = gPopAux;
        gPopAux = tmp;
    }
} Populations;




#endif // __G_POPULATIONS_H