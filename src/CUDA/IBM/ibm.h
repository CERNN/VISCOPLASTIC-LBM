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
#include "structs/particleEulerNodesUpdate.h"
#include "ibmReport.h"
#include "collision/ibmCollision.h"

/**
*   @brief Run immersed boundary method (IBM)
*   
*   @param particles: IBM particles
*   @param macr: macroscopics
*   @param vels_aux: auxiliary vector for velocities
*   @param pop: populations
*   @param gridLBM: LBM CUDA grid size
*   @param threadsLBM: LBM CUDA block size
*   @param streamLBM: LBM CUDA streams for GPUs
*   @param streamIBM: IBM CUDA streams for GPUs
*   @param step: current time step
*   @param pEulerNodes: euler nodes (from LBM) that are used
*/
__host__
void immersedBoundaryMethod(
    ParticlesSoA particles,
    Macroscopics* __restrict__ macr,
    dfloat3SoA* __restrict__ vels_aux,
    Populations* const __restrict__ pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    cudaStream_t streamLBM[N_GPUS],
    cudaStream_t streamIBM[N_GPUS],
    unsigned int step,
    ParticleEulerNodesUpdate* pEulerNodes
);

/**
*   @brief Performs IBM interpolation and spread of force
*   
*   @param particlesNodes: IBM particles nodes
*   @param particleCenters: IBM particles centers
*   @param macr: macroscopics
*   @param velAuxIBM: auxiliary velocity vector
*/
__global__
void gpuForceInterpolationSpread(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    Macroscopics const macr,
    dfloat3SoA velAuxIBM
);


/**
*   @brief Update macroscopics for IBM
*   
*   @param pop: populations
*   @param macr: macroscopics to update
*   @param velAuxIBM: auxiliary velocity vector
*   @param eulerIdxsUpdate: array with euler nodes (from LBM) that must be updated
                (only if IBM_EULER_OPTIMIZATION is true)
*   @param currEulerNodes: number of nodes that must be updated 
                (only if IBM_EULER_OPTIMIZATION is true)
*/
__global__
void gpuUpdateMacrIBM(Populations pop, Macroscopics macr, dfloat3SoA velAuxIBM
    #if IBM_EULER_OPTIMIZATION
    , size_t* eulerIdxsUpdate, unsigned int currEulerNodes
    #endif
);

/**
*   @brief Reset forces from all IBM nodes
*   
*   @param particlesNodes: nodes to reset forces
*/
__global__
void gpuResetNodesForces(ParticleNodeSoA particlesNodes);

/**
*   @brief Copies macroscopics from one GPU border to another
*
*   @param macrBase: macroscopics in GPU with lower z
*   @param macrNext: macroscopics in GPU with higher z
*/
__global__
void copyBorderMacr(Macroscopics macrBase, Macroscopics macrNext);

/**
*   @brief Updated particles velocities and rotation
*   
*   @param particleCenters: particles centers to update
*/
__global__
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

/**
*   @brief Update particles positions
*   
*   @param particleCenters: particles center to update
*/
__global__
void gpuParticleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

/**
*   @brief Update particles nodes positions
*   
*   @param particlesNodes: particles nodes to update
*   @param particleCenters: particles centers
*/
__global__
void gpuParticleNodeMovement(
    ParticleNodeSoA const particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
);


/**
*   @brief Update particles center old values (from last step)
*   
*   @param particleCenters: particles centers to update
*/
__global__
void gpuUpdateParticleOldValues(
    ParticleCenter particleCenters[NUM_PARTICLES]
);



#endif // !__IBM_H