/*
*   @file macrProc.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct to process/treat macroscopics
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __MACR_PROC_H
#define __MACR_PROC_H

#include "../globalFunctions.h"
#include "macroscopics.h"


/*
*   Struct for macroscopics processing in runtime.
*   To evaluate values such as average rho, average velocity in a plan, etc. 
*/
typedef struct macrProc{
    Macroscopics* macrCurr; // current macroscopics to process
    Macroscopics* macrOld;  // old macroscopics if required (as is by residual)
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
        macrOld = nullptr;
        step = nullptr;
        residual = 1;
        avgRho = RHO_0;
        for(int i = 0; i < NY; i++)
            avgUzPlanXZ[i] = 0;
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

}MacrProc;

#endif // !__MACR_PROC_H