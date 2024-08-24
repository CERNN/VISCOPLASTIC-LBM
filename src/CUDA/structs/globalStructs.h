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

/*
*   Struct for dfloat in x, y, z
*/
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

/*
*   Struct for dfloat in x, y, z, w (quartenion)
*/
typedef struct dfloat4{
    dfloat x;
    dfloat y;
    dfloat z;
    dfloat w;

    __host__ __device__
    dfloat4(dfloat x = 0, dfloat y = 0, dfloat z = 0, dfloat w = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
} dfloat4;

/*
*   Struct for dfloat in x, y, z as structure of arrays (SoA)
*/
typedef struct dfloat3SoA {
    int varLocation; // IN_VIRTUAL or IN_HOST
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

    /**
    *   @brief Allocate memory for SoA
    *   
    *   @param arraySize: array size, in number of elements
    *   @param location: array location, IN_VIRTUAL or IN_HOST
    */
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

    /**
    *   @brief Free memory of SoA
    */
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

   

    /**
    *   @brief Copy values from another dfloat3SoA array
    *   
    *   @param arrayRef: arrays to copy values
    *   @param memSize: size of memory to copy, in bytes
    *   @param baseIdx: base index for this
    *   @param baseIdxRef: base index for arrayRef
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

    /**
    *   @brief Copy value from dfloat3
    *   
    *   @param val: dfloat3 to copy values
    *   @param idx: index to write values to
    */
    __host__ __device__
    void copyValuesFromFloat3(dfloat3 val, size_t idx){
        this->x[idx] = val.x;
        this->y[idx] = val.y;
        this->z[idx] = val.z;
    }

    /**
    *   @brief Get the falues from given index
    *   
    *   @param idx: index to copy from
    *   @return dfloat3: dfloat3 with values
    */
    __host__ __device__
    dfloat3 getValuesFromIdx(size_t idx){
        return dfloat3(this->x[idx], this->y[idx], this->z[idx]);
    }

    __host__ __device__
    void leftShift(size_t idx, size_t left_shift){
        this->x[idx-left_shift] = this->x[idx];
        this->y[idx-left_shift] = this->y[idx];
        this->z[idx-left_shift] = this->z[idx];
    }

} dfloat3SoA;

#endif //__GLOBAL_STRUCTS_H
