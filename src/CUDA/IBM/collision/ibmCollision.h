/*
*   @file ibmCollision.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
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
*   @brief Perform particles collisions with each other and walls with soft sphere model
*   
*   @param particleCenters: particles centers to perform colision
*/
__global__ 
void gpuParticlesCollisionSoft(
    ParticleCenter particleCenters[NUM_PARTICLES]
);

/**
*   @brief Perform particles collisions with each other and walls with hard sphere model
*
*   @param particlesNodes: particles nodes to update
*   @param particleCenters: particles centers to perform colision
*/
__global__ 
void gpuParticlesCollisionHard(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
);


#endif // !__IBM_COLLISION_H
