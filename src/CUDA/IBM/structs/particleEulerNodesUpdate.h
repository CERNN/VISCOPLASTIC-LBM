#ifndef __PARTICLE_EULER_NODES_UPDATE_H
#define __PARTICLE_EULER_NODES_UPDATE_H

#include "../ibmVar.h"
#include "particle.h"


/*
*   Struct to describe and update the Eulerian nodes that must update
*   its velocity for IBM
*/
typedef struct particleEulerNodesUpdate{
    // Array as [NZ, NY, NZ] with mask of each node
    uint32_t* eulerMaskArray;
    // Array with indexes values to update
    size_t* eulerIndexesUpdate;
    // Maximum number of euler nodes to update
    unsigned int nEulerNodes;
    // Current number of euler nodes to update
    unsigned int currEulerNodes;

    // Movable particles centers pointers
    ParticleCenter* pCenterMovable;
    // Particles last position
    dfloat3* particlesLastPos;
    // Number of movable particles
    unsigned int nParticlesMovable;

    particleEulerNodesUpdate();
    ~particleEulerNodesUpdate();

    __host__
    void initializeEulerNodes(ParticleCenter p[NUM_PARTICLES]);

    __host__
    void checkParticlesMovement();

    __host__
    void removeUnneededEulerNodes();

    __host__
    void updateEulerNodes(ParticleCenter p, uint32_t mask);

} ParticleEulerNodesUpdate;


__global__
void ibmEulerCopyVelocities(dfloat3SoA dst, dfloat3SoA src, size_t*
    eulerIdxsUpdate, unsigned int currEulerNodes);

#endif // !__PARTICLE_EULER_NODES_UPDATE_H