/**
*   @file globalStructs.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Global general structs
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __GLOBAL_STRUCTS_H
#define __GLOBAL_STRUCTS_H

#include "../var.h"
#include "../errorDef.h"

typedef struct dfloat3 {
    dfloat x;
    dfloat y;
    dfloat z;

    __host__ __device__
    dfloat3(dfloat x = 0, dfloat y = 0, dfloat z = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
} dfloat3;

typedef struct dfloat3SoA {
    int varLocation;
    dfloat* x; // x array
    dfloat* y; // y array
    dfloat* z; // z array

    __host__ __device__
    dfloat3SoA()
    {
        varLocation = 0;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__ __device__
    ~dfloat3SoA()
    {
        varLocation = 0;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__
    void allocateMemory(size_t arraySize, int location = IN_VIRTUAL){
        size_t memSize = sizeof(dfloat) * arraySize;

        this->varLocation = location;
        switch(location){
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->z), memSize));
            break;
        case IN_HOST:
            checkCudaErrors(cudaMallocHost((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->z), memSize));
            break;
        default:
            break;
        }
    }

    __host__
    void freeMemory(){
        switch (this->varLocation)
        {
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->x));
            checkCudaErrors(cudaFree(this->y));
            checkCudaErrors(cudaFree(this->z));
            break;

        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->x));
            checkCudaErrors(cudaFreeHost(this->y));
            checkCudaErrors(cudaFreeHost(this->z));
            break;
        default:
            break;
        }
    }

    /*  
        Copies arrayRef to this object
        this <- arrayRef
        Use for host/device, not virtual
    */
   __host__
    void copyFromDfloat3SoA(dfloat3SoA arrayRef, size_t memSize, size_t baseIdx=0, size_t baseIdxRef=0){

        cudaStream_t streamX, streamY, streamZ;
        checkCudaErrors(cudaStreamCreate(&(streamX)));
        checkCudaErrors(cudaStreamCreate(&(streamY)));
        checkCudaErrors(cudaStreamCreate(&(streamZ)));

        checkCudaErrors(cudaMemcpyAsync(this->x+baseIdx, arrayRef.x+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamX));
        checkCudaErrors(cudaMemcpyAsync(this->y+baseIdx, arrayRef.y+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamY));
        checkCudaErrors(cudaMemcpyAsync(this->z+baseIdx, arrayRef.z+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamZ));

        checkCudaErrors(cudaStreamSynchronize(streamX));
        checkCudaErrors(cudaStreamSynchronize(streamY));
        checkCudaErrors(cudaStreamSynchronize(streamZ));

        checkCudaErrors(cudaStreamDestroy(streamX));
        checkCudaErrors(cudaStreamDestroy(streamY));
        checkCudaErrors(cudaStreamDestroy(streamZ));
    }

    __host__ __device__
    void copyValuesFromdFloat3(dfloat3 val, size_t idx){
        this->x[idx] = val.x;
        this->y[idx] = val.y;
        this->z[idx] = val.z;
    }

    __host__ __device__
    dfloat3 getValuesFromdIdx(size_t idx){
        return dfloat3(this->x[idx], this->y[idx], this->z[idx]);
    }
} dfloat3SoA;

#endif //__GLOBAL_STRUCTS_H
