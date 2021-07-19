/*
*   @file macroscopics.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for LBM macroscopics
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __MACROSCOPICS_H
#define __MACROSCOPICS_H

#include "../var.h"
#include "../globalFunctions.h"
#include "../errorDef.h"
#include "../NNF/nnf.h"
#include "globalStructs.h"
#include <cuda.h>

/*
*   Struct for LBM macroscopics
*/
typedef struct macroscopics
{
private:
    int varLocation;
public:
    dfloat* rho;    // density
    dfloat3SoA u;  // velocity

    #ifdef IBM
    dfloat3SoA f; // force
    #endif

    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega;
    #endif

    /* Constructor */
    __host__
    macroscopics()
    {
        this->rho = nullptr;

        #ifdef NON_NEWTONIAN_FLUID
        this->omega = nullptr;
        #endif
    }

    /* Destructor */
    __host__
    ~macroscopics()
    {
        this->rho = nullptr;

        #ifdef NON_NEWTONIAN_FLUID
        this->omega = nullptr;
        #endif
    }

    /* Allocate macroscopics */
    __host__
    void macrAllocation(int varLocation)
    {
        this->varLocation = varLocation;
        switch (varLocation)
        {
        case IN_HOST:
            // allocate with CUDA for pinned memory and for all GPUS
            checkCudaErrors(cudaMallocHost((void**)&(this->rho), TOTAL_MEM_SIZE_IBM_SCALAR));
            this->u.allocateMemory(TOTAL_NUMBER_LBM_IB_MACR_NODES, IN_HOST);
            #ifdef IBM
            this->f.allocateMemory(TOTAL_NUMBER_LBM_IB_MACR_NODES, IN_HOST);
            #endif
            #ifdef NON_NEWTONIAN_FLUID
            checkCudaErrors(cudaMallocHost((void**)&(this->omega), TOTAL_MEM_SIZE_SCALAR));
            #endif
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->rho), MEM_SIZE_IBM_SCALAR));
            this->u.allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
            #ifdef IBM
            this->f.allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
            #endif
            #ifdef NON_NEWTONIAN_FLUID
            checkCudaErrors(cudaMallocManaged((void**)&(this->omega), MEM_SIZE_SCALAR));
            #endif
            break;
        default:
            break;
        }
    }

    /* Free macroscopics */
    __host__
    void macrFree()
    {
        switch (this->varLocation)
        {
        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->rho));
            this->u.freeMemory();
            #ifdef IBM
            this->f.freeMemory();
            #endif
            #ifdef NON_NEWTONIAN_FLUID
            checkCudaErrors(cudaFreeHost(this->omega));
            #endif
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->rho));
            this->u.freeMemory();
            #ifdef IBM
            this->f.freeMemory();
            #endif
            #ifdef NON_NEWTONIAN_FLUID
            checkCudaErrors(cudaFree(this->omega));
            #endif
            break;
        default:
            break;
        }
    }

    /*  
        Copies macrRef to this object  
        this <- macrRef
    */
    __host__
    void copyMacr(macroscopics* macrRef, size_t baseIdx=0, size_t baseIdxRef=0, bool all_domain=false)
    {
        size_t memSize = (all_domain ? TOTAL_MEM_SIZE_SCALAR : MEM_SIZE_SCALAR);

        cudaStream_t streamRho, streamUx, streamUy, streamUz;
        #if defined(IBM) && EXPORT_FORCES
        cudaStream_t streamFx, streamFy, streamFz;
        #endif
        #ifdef NON_NEWTONIAN_FLUID
        cudaStream_t streamOmega;
        // Constants base index, to use for macroscopics that do not have ghost nodes (omega)
        size_t cteBaseIdx = baseIdx, cteBaseIdxRef = baseIdxRef;
        #endif

        // Sum ghost index of ghost nodes, to not consider it
        if(this->varLocation == IN_VIRTUAL)
            baseIdx += idxScalarWBorder(0, 0, 0);
        if(macrRef->varLocation == IN_VIRTUAL)
            baseIdxRef += idxScalarWBorder(0, 0, 0);

        checkCudaErrors(cudaStreamCreate(&(streamRho)));
        checkCudaErrors(cudaStreamCreate(&(streamUx)));
        checkCudaErrors(cudaStreamCreate(&(streamUy)));
        checkCudaErrors(cudaStreamCreate(&(streamUz)));
        #if defined(IBM) && EXPORT_FORCES
        checkCudaErrors(cudaStreamCreate(&(streamFx)));
        checkCudaErrors(cudaStreamCreate(&(streamFy)));
        checkCudaErrors(cudaStreamCreate(&(streamFz)));
        #endif

        #ifdef NON_NEWTONIAN_FLUID
        checkCudaErrors(cudaStreamCreate(&(streamOmega)));
        #endif

        checkCudaErrors(cudaMemcpyAsync(this->rho+baseIdx, macrRef->rho+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamRho));
        checkCudaErrors(cudaMemcpyAsync(this->u.x+baseIdx, macrRef->u.x+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamUx));
        checkCudaErrors(cudaMemcpyAsync(this->u.y+baseIdx, macrRef->u.y+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamUy));
        checkCudaErrors(cudaMemcpyAsync(this->u.z+baseIdx, macrRef->u.z+baseIdxRef,
            memSize, cudaMemcpyDefault, streamUz));

        #if defined(IBM) && EXPORT_FORCES
        checkCudaErrors(cudaMemcpyAsync(this->f.x+baseIdx, macrRef->f.x+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamFx));
        checkCudaErrors(cudaMemcpyAsync(this->f.y+baseIdx, macrRef->f.y+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamFy));
        checkCudaErrors(cudaMemcpyAsync(this->f.z+baseIdx, macrRef->f.z+baseIdxRef,
            memSize, cudaMemcpyDefault, streamFz));
        #endif

        #ifdef NON_NEWTONIAN_FLUID
        checkCudaErrors(cudaMemcpyAsync(this->omega+cteBaseIdx, macrRef->omega+cteBaseIdxRef,
            memSize, cudaMemcpyDefault, streamOmega));
        #endif

        checkCudaErrors(cudaStreamSynchronize(streamRho));
        checkCudaErrors(cudaStreamSynchronize(streamUx));
        checkCudaErrors(cudaStreamSynchronize(streamUy));
        checkCudaErrors(cudaStreamSynchronize(streamUz));

        checkCudaErrors(cudaStreamDestroy(streamRho));
        checkCudaErrors(cudaStreamDestroy(streamUx));
        checkCudaErrors(cudaStreamDestroy(streamUy));
        checkCudaErrors(cudaStreamDestroy(streamUz));
        #if defined(IBM) && EXPORT_FORCES
        checkCudaErrors(cudaStreamDestroy(streamFx));
        checkCudaErrors(cudaStreamDestroy(streamFy));
        checkCudaErrors(cudaStreamDestroy(streamFz));
        #endif

        #ifdef NON_NEWTONIAN_FLUID
        checkCudaErrors(cudaStreamDestroy(streamOmega));
        #endif

    }

} Macroscopics;


#endif // !__MACROSCOPICS_H