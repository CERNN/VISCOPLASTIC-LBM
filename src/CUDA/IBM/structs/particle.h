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
#include "ibmMacrsAux.h"

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
    ParticleNodeSoA nodesSoA[N_GPUS];
    ParticleCenter* pCenterArray;
    dfloat3* pCenterLastPos;    // Last particle position
    dfloat3* pCenterLastWPos;   // Last angular particle position

    void updateParticlesAsSoA(Particle* particles);
    void updateNodesGPUs();
    void freeNodesAndCenters();
} ParticlesSoA;


/*
*   @brief Create the particle in the shape of a sphere with given diameter and center
*   @param diameter: sphere diameter in dfloat
*   @param center : sphere center position
*   @param coloumb: number of interations for coloumb optimization
*   @param move: particle is movable or not
*   @param density: particle density
*   @param vel: particle velocity
*   @param w: particle rotation velocity
*/
Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,
    dfloat density=PARTICLE_DENSITY, dfloat3 vel=dfloat3(0, 0, 0), dfloat3 w=dfloat3(0, 0, 0));


/*
*   @brief Create a fixed open cylinder in the bases given the diameter and location of the base center
*   @param diameter: cylinder diameter in dfloat
*   @param baseOneCenter: coordinates of the first base center
*   @param baseTwoCenter: coordinates of the second base center
*   @param pattern: false: in-line pattern, true: staggered pattern
*/
Particle makeOpenCylinder(dfloat diameter, dfloat3 baseOneCenter, dfloat3 baseTwoCenter, bool pattern);


#endif