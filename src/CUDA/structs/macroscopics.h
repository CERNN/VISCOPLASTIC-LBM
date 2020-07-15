/*
*   @file macroscopics.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for LBM macroscopics
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __MACROSCOPICS_H
#define __MACROSCOPICS_H

#include "..\var.h"
#include "..\globalFunctions.h"
#include "..\errorDef.h"
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
    dfloat* ux;     // x velocity
    dfloat* uy;     // y velocity
    dfloat* uz;     // z velocity

    /* Constructor */
    __host__
    macroscopics()
    {
        this->rho = nullptr;
        this->ux = nullptr;
        this->uy = nullptr;
        this->uz = nullptr;
    }

    /* Destructor */
    __host__
    ~macroscopics()
    {
        this->rho = nullptr;
        this->ux = nullptr;
        this->uy = nullptr;
        this->uz = nullptr;
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
            checkCudaErrors(cudaMallocHost((void**)&(this->rho), totalMemSizeScalar));
            checkCudaErrors(cudaMallocHost((void**)&(this->ux), totalMemSizeScalar));
            checkCudaErrors(cudaMallocHost((void**)&(this->uy), totalMemSizeScalar));
            checkCudaErrors(cudaMallocHost((void**)&(this->uz), totalMemSizeScalar));
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->rho), memSizeScalar));
            checkCudaErrors(cudaMallocManaged((void**)&(this->ux), memSizeScalar));
            checkCudaErrors(cudaMallocManaged((void**)&(this->uy), memSizeScalar));
            checkCudaErrors(cudaMallocManaged((void**)&(this->uz), memSizeScalar));
            break;
        default:
            break;
        }
    }

    /* Free macroscopics */
    __host__
    void macrFree()
    {
        switch (varLocation)
        {
        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->rho));
            checkCudaErrors(cudaFreeHost(this->ux));
            checkCudaErrors(cudaFreeHost(this->uy));
            checkCudaErrors(cudaFreeHost(this->uz));
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->rho));
            checkCudaErrors(cudaFree(this->ux));
            checkCudaErrors(cudaFree(this->uy));
            checkCudaErrors(cudaFree(this->uz));
            break;
        default:
            break;
        }
    }

    /*  
        Copies macrRef to this object  
        this <- macrRef
        TODO: update to Memcpy 
    */
    __host__
    void copyMacr(macroscopics* macrRef, size_t baseIdx=0, bool all_domain=false)
    {
        int total_z = (all_domain ? NZ_TOTAL : NZ);
        for(int z = 0; z < total_z; z++)
            for (unsigned int y = 0; y < NY; y++)
                for (unsigned int x = 0; x < NX; x++)
                {
                    size_t idx = idxScalar(x, y, z);
                    this->rho[idx+baseIdx] = macrRef->rho[idx];
                    this->ux[idx+baseIdx] = macrRef->ux[idx];
                    this->uy[idx+baseIdx] = macrRef->uy[idx];
                    this->uz[idx+baseIdx] = macrRef->uz[idx];
                }
    }

} Macroscopics;


#endif // !__MACROSCOPICS_H