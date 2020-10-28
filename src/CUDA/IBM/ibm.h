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

#include <string>
#include "ibmGlobalFunctions.h"
#include "../lbm.h"
#include "../structs/macroscopics.h"
#include "../structs/populations.h"
#include "../structs/globalStructs.h"
#include "structs/particle.h"
#include "ibmReport.h"


__host__
void immersedBoundaryMethod(
    ParticlesSoA particles,
    Macroscopics* __restrict__ macr,
    dfloat3SoA* __restrict__ vels_aux,
    Populations* const __restrict__ pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    unsigned int gridIBM,
    unsigned int threadsIBM,
    cudaStream_t __restrict__ streamLBM[N_GPUS],
    cudaStream_t __restrict__ streamIBM[N_GPUS],
    unsigned int step
);


__global__
void gpuForceInterpolationSpread(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    Macroscopics const macr,
    dfloat3SoA velAuxIBM
);


__global__
void gpuUpdateMacrResetForces(Populations pop, Macroscopics macr, dfloat3SoA velAuxIBM);


__global__
void gpuResetNodesForces(ParticleNodeSoA particlesNodes);


__global__
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

__global__
void gpuParticleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

__global__
void gpuUpdateParticleOldValues(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

__global__
void gpuParticleNodeMovement(
    ParticleNodeSoA const particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
);

__global__ 
void gpuParticlesCollision(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

#endif // !__IBM_H