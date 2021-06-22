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
#include "../ibmBoundaryCondition.h"

#include "../ibmGlobalFunctions.h"
#include "../../structs/globalStructs.h"
#include "../structs/particle.h"



/**
*   @brief Perform particles collisions combination, and with the walls.
*
*   @param particleCenters: particles centers to perform colision
*   @param step: current time step
*/
__global__ 
void gpuParticlesCollision(

    ParticleCenter particleCenters[NUM_PARTICLES]
    unsigned int step

);

/**
*   @brief Perform particles collisions with wall using soft sphere collision model
*
*   @param displacement: total normal displacement
*   @param wallNormalVector: wall normal vector  
*   @param particleCenter: particles centers to perform colision index i
*   @param step: current time step
*/
__device__ 
void gpuSoftSphereWallCollision(
    dfloat displacement,
    dfloat3 wallNormalVector,
    ParticleCenter* pc_i,
    unsigned int step
);

/**
*   @brief Perform particles collisions with other particles using soft sphere collision model
*
*   @param displacement: total normal displacement
*   @param column: particle i index
*   @param row: particle j index
*   @param particleCenter: particles centers to perform colision index i
*   @param particleCenter: particles centers to perform colision index j
*   @param step: current time step
*/
__device__ 
void gpuSoftSphereParticleCollision(
    dfloat displacement,
    unsigned int column,
    unsigned int row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    unsigned int step
);

/**
*   @brief Perform collision displacement tracker between particle and wall
*   
*   @param n: wall normal vector  
*   @param pc_i: particles centers to perform colision
*   @param step: current time step
*/
__device__
int gpuTangentialDisplacementTrackerWall(
    dfloat3 n,
    ParticleCenter* pc_i,
    unsigned int step
);


/**
*   @brief Perform collision displacement tracker between particles
*   
*   @param column: particle i index
*   @param row: particle j index
*   @param pc_i: particles centers to perform colision
*   @param pc_j: particles centers to perform colision
*   @param step: current time step
*/
__device__
int gpuTangentialDisplacementTrackerParticle(
    unsigned int column,
    unsigned int row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    unsigned int step
);




#if defined LUBRICATION_FORCE
/**
*   @brief Perform lubrication force between wall and particle
*
*   @param displacement: total normal displacement
*   @param wallNormalVector: wall normal vector  
*   @param particleCenter: particles centers to perform colision index i
*/
__device__ 
void gpuLubricationWall(
    dfloat gap,
    dfloat3 wallNormalVector,
    ParticleCenter* pc_i
);

/**
*   @brief Perform  the lubrication force between particles
*   @param displacement: total normal displacement
*   @param particleCenter: particles centers to perform colision index i
*   @param particleCenter: particles centers to perform colision index j
*/
__device__ 
void gpuLubricationParticle(
    dfloat gap,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j
);


#endif // !__IBM_COLLISION_H
