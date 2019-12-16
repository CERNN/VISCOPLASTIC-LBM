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
*   Struct for boundary conditions info
*/
typedef struct boundaryConditionsInfo{
    // Number of boundary conditions nodes
    size_t totalBCNodes;
    // Number of non local boundary conditions nodes
    size_t totalNonLocalBCNodes;
    // Index of non local boundary conditions nodes
    size_t* idxBCNodes;

    /* Constructor */
    __host__
    boundaryConditionsInfo()
    {
        totalBCNodes = 0;
        totalNonLocalBCNodes = 0;
        idxBCNodes = nullptr;
    }

    /* Destructor */
    __host__
    ~boundaryConditionsInfo()
    {
        totalBCNodes = 0;
        totalNonLocalBCNodes = 0;
        idxBCNodes = nullptr;
    }

    __host__
    void allocateIdxBC()
    {
        if(totalBCNodes <= 0)
            return;
        size_t memSizeIdxBC = totalBCNodes*sizeof(size_t);
        checkCudaErrors(cudaMallocManaged((void**)&(this->idxBCNodes), memSizeIdxBC));
    }

    __host__
    void freeIdxBC()
    {
        if(idxBCNodes == nullptr)
            return;
        checkCudaErrors(cudaFree(this->idxBCNodes));
        idxBCNodes = nullptr;
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

        if(totalBCNodes <= 0)
            return;

        // allocate memory for idx
        allocateIdxBC();

        // update index of non local boundary conditions
        int i = 0;
        for(int z = 0; z < NZ; z++)
            for(int y = 0; y < NY; y++)
                for(int x = 0; x < NX; x++)
                {
                    NodeTypeMap ntm = mapBC[idxScalar(x, y, z)];
                    if(ntm.getIsUsed())
                        if(ntm.getSchemeBC() != BC_NULL)
                        {
                            idxBCNodes[i] = idxScalar(x, y, z);
                            i++;
                        }
                }
    }

}BoundaryConditionsInfo;

#endif // !__BOUNDARY_CONDITIONS_INFO_H