#ifndef __IBM_MACRS_AUX_H
#define __IBM_MACRS_AUX_H

#include "../ibmVar.h"
#include "../../globalFunctions.h"

/*
*   Struct for LBM macroscopics
*/
typedef struct ibmMacrsAux
{
public:
    dfloat3SoA velAux[N_GPUS];  // auxiliary velocities
    dfloat3SoA fAux[N_GPUS]; // auxiliary forces, for synchronization

    /* Constructor */
    __host__
    ibmMacrsAux()
    {
    }

    /* Destructor */
    __host__
    ~ibmMacrsAux()
    {
    }

    /* Allocate IBM macroscopics aux */
    __host__
    void ibmMacrsAuxAllocation()
    {
        for(int i = 0; i < N_GPUS; i++)
        {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            this->velAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
            this->fAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
        }
    }

    /* Free IBM macroscopics aux */
    __host__
    void ibmMacrsAuxFree()
    {
        for(int i = 0; i < N_GPUS; i++)
        {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            this->velAux[i].freeMemory();
            this->fAux[i].freeMemory();
        }
    }
} IBMMacrsAux;

#endif // !__IBM_MACRS_AUX_H