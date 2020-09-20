/*
*   @file ibm.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief IBM steps: perform interpolation and spread force
*   @version 0.3.0
*   @date 26/08/2020
*/


#ifndef __IBM_H
#define __IBM_H

#include "ibmVar.h"

#include "ibmGlobalFunctions.h"
#include "../structs/macroscopics.h"
#include "../structs/populations.h"
#include "../structs/globalStructs.h"
#include "structs/particleCenter.h"
#include "structs/particleNode.h"


__host__
void immersedBoundaryMethod(
    Particle* const __restrict__ particles,
    Macroscopics* const __restrict__ macr,
    Populations* const __restrict__ pop,
    const dim3 gridLBM,
    const dim3 threadsLBM,
    const unsigned int gridIBM,
    const unsigned int threadsIBM,
    cudaStream_t* const __restrict__ stream
);


__global__
void gpuForceInterpolationSpread(
    unsigned int totalParticlesNodes,
    ParticleNode* const __restrict__ particlesNodes,
    Macroscopics* const __restrict__ macr
);


__global__
void gpuUpdateMacrResetForces(Populations* __restrict__ pop, Macroscopics* __restrict__ macr);


__global__
void gpuResetNodesForces(ParticleNodeSoA* __restrict__ particleNodes);


__host__ 
void updateParticleCenterForce(
    Particle* __restrict__ particleNode,
    unsigned int numParticles
);

__host__ 
dfloat3 particleCollisionSoft(
    ParticleCenter* __restrict__ particleCenter,
    int particleIndex
);

__global__
void particleMovement(
    ParticleCenter* __restrict__ ParticleCenter
);

#endif // !__IBM_H