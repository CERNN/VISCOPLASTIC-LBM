#ifndef __PARTICLE_EULER_NODES_UPDATE_H
#define __PARTICLE_EULER_NODES_UPDATE_H

#include "../ibmVar.h"
#include "particle.h"
#include "../../structs/macroscopics.h"
#include "../../globalFunctions.h"

/*
*   Struct to describe and update the Eulerian nodes that must update
*   its velocity for IBM
*/
typedef struct particleEulerNodesUpdate{
    // Array as [NZ, NY, NZ] with mask of each node for each GPU
    uint32_t* eulerMaskArray[N_GPUS];
    // Array with indexes values to update for each GPU
    size_t* eulerIndexesUpdate[N_GPUS];
    // Maximum number of euler nodes to update
    unsigned int maxEulerNodes;
    // Current number of euler nodes to update for each GPU
    unsigned int currEulerNodes[N_GPUS];
    // Number of euler nodes that are fixed for each GPU
    unsigned int eulerFixedNodes[N_GPUS];

    // Movable particles centers pointers
    ParticleCenter** pCenterMovable;
    // Particles last position
    dfloat3* particlesLastPos;
    // Particles last angular position
    dfloat3* particlesLastWPos;
    // Has fixed particles
    bool hasFixed;
    // TODO: keep particle angle as well to count rotation
    // Number of movable particles
    unsigned int nParticlesMovable;

    particleEulerNodesUpdate();
    ~particleEulerNodesUpdate();

    /**
    *   @brief Initialize euler nodes that must be updated, given particles
    *   
    *   @param p: simulation particles
    */
    __host__
    void initializeEulerNodes(ParticleCenter p[NUM_PARTICLES]);

    /**
    *   @brief Free allocated variables
    */
    __host__
    void freeEulerNodes();

    /**
    *   @brief Check if any particle has moved and update nodes if required
    */
    __host__
    void checkParticlesMovement();

    /**
    *   @brief Remove euler nodes that don't need to be updated anymore
    *   
    *   @param maskRemove: mask with bits to remove with 1 and others as 0
    *   @param n_gpu: gpu number to remove nodes
    */
    __host__
    void removeUnneededEulerNodes(uint32_t maskRemove, int n_gpu);

    /**
    *   @brief Update Euler nodes that must be updated, adding nodes from particle
    *   
    *   @param p: particle to consider nodes
    *   @param mask: mask to use while updating nodes
    *   @param n_gpu: gpu number to update
    */
    __host__
    unsigned int updateEulerNodes(ParticleCenter* p, uint32_t mask, int n_gpu);

} ParticleEulerNodesUpdate;

/**
*   @brief Sum values from auxiliary IBM macroscopics to "main" macroscopics, 
*       given array of indexes to copy, then reset aux values
*   
*   @param dst: destiny dfloat3 array (as SoA) with shape [NZ, NY, NX]
*   @param ibmMacrsAux: Auxiliary macroscopics to sum and reset
*   @param eulerIdxsUpdate: array with indexes to update
*   @param currEulerNodes: array with indexes size
*   @param n_gpu: current GPU number
*/
__global__
void gpuEulerSumIBMAuxsReset(Macroscopics macr, IBMMacrsAux ibmMacrsAux, 
    #if IBM_EULER_OPTIMIZATION
    size_t* eulerIdxsUpdate, unsigned int currEulerNodes, 
    #endif
    int n_gpu);

#endif // !__PARTICLE_EULER_NODES_UPDATE_H