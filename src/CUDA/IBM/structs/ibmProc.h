#ifndef __IBM_PROC_H
#define __IBM_PROC_H

#include "../ibmVar.h"
#include "../../globalFunctions.h"
#include "particle.h"
#include "../../structs/macroscopics.h"
#include "../../lbmReport.h"

/*
*   Struct for IBM post processing
*/
typedef struct ibmProc{
    Macroscopics* macrCurr;  // current macroscopics values
    int* step;               // pointer to step value

    // Treated values below
    dfloat reynolds;
    // Drag coefficient (particle moving in z direction)
    dfloat cd;
    // Lift coefficient in x
    dfloat clx;
    // Lift coefficient in y
    dfloat cly;
    // Particle position
    dfloat3 pos;
    // Particle velocity
    dfloat3 vel;

    /* Constructor */
    __host__
    ibmProc(){
        this->step = nullptr;
        this->macrCurr = nullptr;
        reynolds = 0;
        cd = 0;
        clx = 0;
        cly = 0;
    }

    /* Destructor */
    __host__
    ~ibmProc(){
        this->step = nullptr;
        this->macrCurr = nullptr;
    }

}IBMProc;

#endif // !__IBM_PROC_H