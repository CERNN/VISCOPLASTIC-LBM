/*
*   @file ibmStruct.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for IBM particle node
*   @version 0.3.0
*   @date 16/09/2020
*/

#ifndef __PARTICLE_NODE_H
#define __PARTICLE_NODE_H


#include "../ibmVar.h"
#include "../../structs/globalStructs.h"

/*
*   Describe the IBM node properties
*/
typedef struct particleNode {
    dfloat3 pos; // node coordinate
    dfloat3 vel; // node velocity
    dfloat3 vel_old;
    dfloat3 f;  // node force
    dfloat3 deltaF;  // node force variation
    dfloat S; // node surface area
} ParticleNode;

/*
*   Struct to represent the particle nodes as a Structure of Arrays, 
*   instead of a Array of Structures
*/
typedef struct particleNodeSoA {
    unsigned int numNodes; // number of nodes
    unsigned int* particleCenterIdx; // index of particle center for each node

    dfloat3SoA pos; // vectors with nodes coordinates
    dfloat3SoA vel; // vectors with nodes velocities
    dfloat3SoA vel_old; // vectors with nodes old velocities
    dfloat3SoA f;  // vectors with nodes forces
    dfloat3SoA deltaF;  // vectors with nodes forces variations
    dfloat* S; // vector node surface area

    particleNodeSoA();
    ~particleNodeSoA();

    void allocateMemory(unsigned int numNodes);
    void freeMemory();
    void copyNodesFromParticle(struct particle p, unsigned int pCenterIdx, unsigned int baseIdx);

} ParticleNodeSoA;

#endif // !__PARTICLE_NODE_H