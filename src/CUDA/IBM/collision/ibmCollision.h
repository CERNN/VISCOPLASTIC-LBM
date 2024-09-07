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



//collision tracking
/**
*   @brief Calculate the index for wall collisions based on the normal vector.
*   @param n: The normal vector of the wall.
*   @return The calculated index used to identify wall collisions.
*/
__device__ 
int calculateWallIndex(const dfloat3 &n);
/**
*   @brief Find the index of the collision record for a given partnerID.
*   @param collisionData: The data structure containing collision information.
*   @param partnerID: The ID of the collision partner to search for.
*   @param currentTimeStep: The current time step to check collision validity.
*   @return The index of the collision record if found, otherwise -1.
*/
__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep);
/**
*   @brief Start a new collision record for a given partnerID.
*   @param collisionData: The data structure to store collision information.
*   @param partnerID: The ID of the collision partner.
*   @param isWall: Boolean indicating whether the collision involves a wall.
*   @param wallNormal: The normal vector of the wall (if isWall is true).
*   @param currentTimeStep: The current time step to record the collision.
*   @return The index of the newly created collision record or -1 if none available.
*/
__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep);
/**
*   @brief Update the tangential displacement and collision step time for a collision record.
*   @param collisionData: The data structure containing collision information.
*   @param index: The index of the collision record to update.
*   @param displacement: The displacement to add to the tangential displacement.
*   @param currentTimeStep: The current time step to update the last collision step.
*   @return The updated tangential displacement for the collision record.
*/
__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, int currentTimeStep);
/**
*   @brief End a collision record and reset its data if necessary.
*   @param collisionData: The data structure containing collision information.
*   @param index: The index of the collision record to end.
*   @param currentTimeStep: The current time step to check the validity of ending the collision.
*/
__device__ 
void endCollision(CollisionData &collisionData, int index, int currentTimeStep);


//collision with wall

/**
*   @brief Check for collisions between a particle and walls based on the particle's shape.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing particle information.
*   @param step: The current time step for collision checking.
*   This function determines the type of the particle (sphere, capsule, or ellipsoid) and
*   calls the appropriate function to check for collisions with walls based on the particle's shape.
*   If the shape is unknown, no action is taken.
*/
__device__
void checkCollisionWalls(ParticleCenter* pc_i, unsigned int step);
/**
*   @brief Check for collisions between a sphere and walls.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
*   @param step: The current time step for collision checking.
*   This function checks for collisions between a sphere and walls. It uses the particle's
*   properties (such as position and radius) to determine if there is an intersection with any walls.
*/
__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i, unsigned int step);



//collision mechanics with walls

/**
*   @brief Handle collision mechanics between a sphere and a wall.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
*   @param wallData: The data structure representing the wall.
*   @param displacement: The displacement value representing how far the sphere has moved.
*   @param step: The current time step for collision processing.
*   This function calculates and processes the collision between a sphere and a wall. It uses
*   the sphere's position and displacement to determine and handle the interaction with the wall.
*/
__device__
void sphereWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,int step);

//sphere functions
/**
*   @brief Compute the gap between two spheres.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
*   @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
*   @return The distance between the surfaces of the two spheres.
*   This function calculates the gap between two spheres based on their positions and radii. The result is
*   the distance between the surfaces of the two spheres, which can be used to determine if a collision has occurred.
*/
__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j);

//capsule functions

//ellipsoid functions
// collision between particles themselves
/**
*   @brief Check for collisions between two particles by comparing the pair types and calling the proper function.
*   @param column: The column index in a grid or matrix representing the particle's position.
*   @param row: The row index in a grid or matrix representing the particle's position.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first particle.
*   @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second particle.
*   @param step: The current time step for collision checking.
*/
__device__
void checkCollisionBetweenParticles(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step);


/**
*   @brief Handle collision mechanics between two spheres.
*   @param column: The column index in a grid or matrix representing the particles' positions.
*   @param row: The row index in a grid or matrix representing the particles' positions.
*   @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
*   @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
*   @param step: The current time step for collision processing.
*   This function calculates and processes collisions between two spheres based on their positions, radii, and properties.
*/
__device__
void sphereSphereCollision(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step);

//collission 

/**
*   @brief Handles collisions between particles and walls or between pairs of particles on the GPU.
*   @param particleCenters: Array of `ParticleCenter` structures representing all particles.
*   @param step: The current time step for collision processing.
*   This function maps a 1D array of particles to a Floyd triangle structure. It checks collisions 
*   between particles in pairs based on their indices and checks particle-wall collisions for the last row.
*   
*   The function calculates the row and column indices from the linear index using the properties 
*   of the Floyd triangle and then determines whether to check for collisions between particles 
*   or between a particle and a wall based on the row index.
*   
*   For particle-pair collisions, it compares particles in the column and row positions.
*   For wall collisions, it compares the particle in the column with the wall.
*/
__global__
void gpuParticlesCollisionHandler(ParticleCenter particleCenters[NUM_PARTICLES], unsigned int step);





/**
*   @brief Perform particles collisions combination, and with the walls.
*
*   @param particleCenters: particles centers to perform colision
*   @param step: current time step
*/
__global__ 
void gpuParticlesCollision(
    ParticleCenter particleCenters[NUM_PARTICLES],
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
#endif //LUBRICATION_FORCE

#endif // !__IBM_COLLISION_H
