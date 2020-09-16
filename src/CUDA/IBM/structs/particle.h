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
#include "ibmVar.h"

struct particle {
    // variables
    unsigned int numNodes; // number of nodes the particle have
    ParticleCenter bodyCenter;
    ParticleNode* nodes;

    Particle(){
        numNodes = 0;
        nodes = nullptr;
    }
} Particle;


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