/*
*   @file ibmCollision.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief IBM Collision: perform particle collision
*   @version 0.3.0
*   @date 15/02/2021
*/

#ifndef __IBM_COLLISION_H
#define __IBM_COLLISION_H

#include "../ibmVar.h"

#include "../ibmGlobalFunctions.h"
#include "../../structs/globalStructs.h"
#include "../structs/particle.h"



/**
*   @brief Perform particles collisions combination, and with the walls.
*   
*   @param particlesNodes: particles nodes to update
*   @param particleCenters: particles centers to perform colision
*/
__global__ 
void gpuParticlesCollision(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
);

/**
*   @brief Perform particles collisions with wall using soft sphere collision model
*
*   @param displacement: total normal displacement
*   @param wallNormalVector: wall normal vector  
*   @param particleCenter: particles centers to perform colision index i
*/
__device__ 
void gpuSoftSphereWallCollision(
    dfloat displacement,
    dfloat3 wallNormalVector,
    ParticleCenter* pc_i
);

/**
*   @brief Perform particles collisions with other particles using soft sphere collision model
*
*   @param displacement: total normal displacement
*   @param particleCenter: particles centers to perform colision index i
*   @param particleCenter: particles centers to perform colision index j
*/
__device__ 
void gpuSoftSphereParticleCollision(
    dfloat displacement,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j
);


/**
*   @brief Perform particles collisions with wall using soft sphere collision model
*
*   @param column: particle i index
*   @param penetration: total penetration
*   @param n: wall normal vector  
*   @param particleCenter: particles centers to perform colision
*   @param particlesNodes: particles nodes to update
*/
__device__ 
void gpuHardSphereWallCollision(
    dfloat column,
    dfloat3 penetration,
    dfloat3 n,
    ParticleCenter* pc_i,
    ParticleNodeSoA particlesNodes
);

/**
*   @brief Perform particles collisions with other particles using soft sphere collision model
*   
*   @param column: particle i index
*   @param row: particle j index
*   @param particleCenter: particles centers to perform colision index i
*   @param particleCenter: particles centers to perform colision index j
*   @param particlesNodes: particles nodes to update
*/
__device__ 
void gpuHarSpheredParticleCollision(
    dfloat column,
    dfloat row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    ParticleNodeSoA particlesNodes
);


#endif // !__IBM_COLLISION_H