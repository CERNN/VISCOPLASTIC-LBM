#ifndef __IBM_PROC_H
#define __IBM_PROC_H

#include "../globalFunctions.h"
#include "../../structs/macroscopics.h"

typedef struct ibmProc{
    Macroscopics* macrCurr; // current macroscopics values
    int* step;              // pointer to step value 

    // Treated values below
    dfloat residual;
    dfloat avgRho;
    dfloat avgUzPlanXZ[NY]; // average Uz velocity in all XZ plans

    /* Constructor */
    __host__
    macrProc()
    {
        macrCurr = nullptr;
        step = nullptr;
    }

    /* Destructor */
    __host__
    ~macrProc()
    {
        macrCurr = nullptr;
        macrOld = nullptr;
        step = nullptr;
        residual = 0;
        avgRho = 0;
    }

    /* Allocate necessary variables, if required dynamic allocation */
    __host__
    void allocateMacrProc()
    {
    }

    /* Free allocated variables, if required dynamic allocation */
    __host__
    void freeMacrProc()
    {
    }

}IBMProc;

#endif // !__IBM_PROC_H