/**
*   @file particle.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for IBM particle
*   @version 0.3.0
*   @date 26/09/2020
*/

#ifndef __PARTICLE_H
#define __PARTICLE_H

#include "../../structs/globalStructs.h"
#include "particleCenter.h"
#include "particleNode.h"
#include "../ibmVar.h"

// After use_math_defines in "var.h"
#include <math.h>

/*
*   Struct for particle representation
*/
typedef struct particle {
    unsigned int numNodes; // number of nodes of particle
    ParticleCenter pCenter; // Particle center
    ParticleNode* nodes; // Particle nodes

    particle(){
        numNodes = 0;
        nodes = nullptr;
    }
} Particle;


/*
*   Particles representation as Struct of arrays (SoA) for better GPU performance
*/
typedef struct particlesSoA{
    ParticleNodeSoA nodesSoA;
    ParticleCenter* pCenterArray;

    void updateParticlesAsSoA(Particle* particles);
    void freeNodesAndCenters();
} ParticlesSoA;


/*
*   @brief Create the particle in the shape of a sphere with given diameter and center
*   @param diameter: sphere diameter in dfloat
*   @param center : sphere center position
*   @param coloumb: number of interations for coloumb optimization
*   @param move: particle is movable or not
*/
Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move);

// Particle makeCylinder(dfloat diameter, dfloat3 begin, dfloat3 end, bool hexa);

#endif