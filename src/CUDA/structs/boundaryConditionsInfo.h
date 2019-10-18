/*
*   @file boundaryConditionsInfo.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct to provide information about the boundary conditions
*   @version 0.2.0
*   @date 17/10/2019
*/

#ifndef __BOUNDARY_CONDITIONS_INFO_H
#define __BOUNDARY_CONDITIONS_INFO_H
#include "../var.h"
#include "../globalFunctions.h"
#include "../errorDef.h"
#include "nodeTypeMap.h"
#include <cuda.h>

/* 
*   Struct for boundary conditions info, used mainly to apply non local 
*   boundary conditions
*/
typedef struct boundaryConditionsInfo{
    // Number of boundary conditions nodes
    size_t totalBCNodes;
    // Number of non local boundary conditions nodes
    size_t totalNonLocalBCNodes;
    // Index of non local boundary conditions nodes
    size_t* idxNonLocalBCNodes;

    /* Constructor */
    __host__
    boundaryConditionsInfo()
    {
        totalBCNodes = 0;
        totalNonLocalBCNodes = 0;
        idxNonLocalBCNodes = nullptr;
    }

    /* Destructor */
    __host__
    ~boundaryConditionsInfo()
    {
        totalBCNodes = 0;
        totalNonLocalBCNodes = 0;
        idxNonLocalBCNodes = nullptr;
    }

    __host__
    void allocateIdxNonLocal()
    {
        if(totalNonLocalBCNodes <= 0)
            return;
        size_t memSizeIdxNonLocal = totalNonLocalBCNodes*sizeof(size_t);
        checkCudaErrors(cudaMallocManaged((void**)&(this->idxNonLocalBCNodes), memSizeIdxNonLocal));
    }

    __host__
    void freeIdxNonLocal()
    {
        if(idxNonLocalBCNodes == nullptr)
            return;
        checkCudaErrors(cudaFree(this->idxNonLocalBCNodes));
    }

    __host__
    void setupBoundaryConditionsInfo(NodeTypeMap* mapBC)
    {
        totalBCNodes = 0;
        totalNonLocalBCNodes = 0;

        // get number of BC nodes
        for(int z = 0; z < NZ; z++)
            for(int y = 0; y < NY; y++)
                for(int x = 0; x < NX; x++)
                {
                    NodeTypeMap ntm = mapBC[idxScalar(x, y, z)];
                    if(ntm.getIsUsed())
                        if(ntm.getSchemeBC() != BC_NULL)
                        {
                            totalBCNodes++;
                            if(!(ntm.isBCLocal()))
                                totalNonLocalBCNodes++;
                        }
                }

        if(totalNonLocalBCNodes <= 0)
            return;

        // allocate memory for idx
        allocateIdxNonLocal();

        // update index of non local boundary conditions
        int i = 0;
        for(int z = 0; z < NZ; z++)
            for(int y = 0; y < NY; y++)
                for(int x = 0; x < NX; x++)
                {
                    NodeTypeMap ntm = mapBC[idxScalar(x, y, z)];
                    if(ntm.getIsUsed())
                        if(ntm.getSchemeBC() != BC_NULL)
                            if(!(ntm.isBCLocal()))
                            {
                                idxNonLocalBCNodes[i] = idxScalar(x, y, z);
                                i++;
                            }
                }
    }

    __host__ __device__
    bool hasNonLocalBC()
    {
        return totalNonLocalBCNodes > 0;
    }

}BoundaryConditionsInfo;

#endif // !__BOUNDARY_CONDITIONS_INFO_H