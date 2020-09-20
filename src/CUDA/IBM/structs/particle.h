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

class Particle {
// Use as struct, as imperative programming
public: 
    unsigned int numNodes; // number of nodes the particle have
    // TODO: rename to pCenter?
    ParticleCenter bodyCenter; // Particle center
    ParticleNode* nodes; // Particle nodes

    // All particle nodes from all particles as struct of arrays (better 
    // performance on GPU)
    static ParticleNodeSoA nodesSoA;

    // Array with particle centers from all particles 
    // (managed for use in CPU and GPU)
    __managed__ static ParticleCenter pCenterArray[NUM_PARTICLES];

    Particle(){
        numNodes = 0;
        nodes = nullptr;
    }

    static void updateParticleStaticVars(Particle* particle);
};


/*
*   @brief Create the particle in the shape of a sphere with given diameter and center
*   @param diameter: sphere diameter in dfloat
*   @param center : sphere center position
*   @param coloumb: number of interations for coloumb optimization
*   @param move: particle is movable or not
*/
Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move);

Particle makeCylinder(dfloat diameter, dfloat3 begin, dfloat3 end, bool hexa);

#endif