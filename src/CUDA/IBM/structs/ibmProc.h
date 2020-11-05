#ifndef __IBM_PROC_H
#define __IBM_PROC_H

#include "../../globalFunctions.h"
#include "../../structs/macroscopics.h"

typedef struct ibmProc{
    Macroscopics* macrCurr;  // current macroscopics values
    ParticleCenter* pCenter; // current particle centers values
    int* step;               // pointer to step value 

    // Treated values below

    /* Constructor */
    __host__
    ibmProc()
    {
        macrCurr = nullptr;
        pCenter = nullptr;
        step = nullptr;
    }

    /* Destructor */
    __host__
    ~ibmProc()
    {
        macrCurr = nullptr;
        pCenter = nullptr;
        step = nullptr;
    }

    /* Allocate necessary variables, if required dynamic allocation */
    __host__
    void allocateIBMProc()
    {
    }

    /* Free allocated variables, if required dynamic allocation */
    __host__
    void freeIBMProc()
    {
    }
}IBMProc;

#endif // !__IBM_PROC_H