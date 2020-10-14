#ifndef __PARTICLE_PROC_H
#define __PARTICLE_PROC_H

#include "particle.h"


typedef struct particleProc{
    int* step;              // Pointer to step value 

    /* Constructor */
    __host__
    particleProc()
    {
        step = nullptr;
    }

    /* Destructor */
    __host__
    ~particleProc()
    {
        step = nullptr;
    }

    /* Allocate necessary variables, if required dynamic allocation */
    __host__
    void allocateParticleProc()
    {
    }

    /* Free allocated variables, if required dynamic allocation */
    __host__
    void freeParticleProc()
    {
    }

}ParticleProc;

#endif // !__PARTICLE_PROC_H