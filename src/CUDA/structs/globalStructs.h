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
    dfloat3()
    {
        x = 0;
        y = 0;
        z = 0;
    }
} dfloat3;

typedef struct dfloat3SoA {
    dfloat* x; // x array
    dfloat* y; // y array
    dfloat* z; // z array

    __host__ __device__
    dfloat3SoA()
    {
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__ __device__
    ~dfloat3SoA()
    {
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__
    void allocateMemory(size_t arraySize){
        size_t memSize = sizeof(dfloat) * arraySize;

        checkCudaErrors(cudaMallocManaged((void**)&(this->x), memSize));
        checkCudaErrors(cudaMallocManaged((void**)&(this->y), memSize));
        checkCudaErrors(cudaMallocManaged((void**)&(this->z), memSize));
    }

    __host__
    void freeMemory(){
        checkCudaErrors(cudaFree(this->x));
        checkCudaErrors(cudaFree(this->y));
        checkCudaErrors(cudaFree(this->z));
    }

    __host__ __device__
    void copyValuesFromdFloat3(dfloat3 val, size_t idx){
        this->x[idx] = val.x;
        this->y[idx] = val.y;
        this->z[idx] = val.z;
    }

} dfloat3SoA;

#endif //__GLOBAL_STRUCTS_H
