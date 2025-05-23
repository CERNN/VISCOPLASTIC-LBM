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

    /*
    *   @brief Create the particle in the shape of a sphere with given diameter and center
    *   @param part: particle object to override values
    *   @param axes: sphere diameter in dfloat
    *   @param center : sphere center position
    *   @param iterations: number of interations for remeshing
    *   @param move: particle is movable or not
    *   @param density: particle density
    *   @param vel: particle velocity
    *   @param w: particle rotation velocity
    */
    __host__
    void makeSphereIco(dfloat diameter, dfloat3 center, bool move,
        dfloat density = PARTICLE_DENSITY, dfloat3 vel = dfloat3(0, 0, 0), dfloat3 w = dfloat3(0, 0, 0));

    /*
    *   @brief Create the particle in the shape of a sphere with given diameter and center
    *   @param part: particle object to override values
    *   @param diameter: sphere diameter in dfloat
    *   @param center : sphere center position
    *   @param coloumb: number of interations for coloumb optimization
    *   @param move: particle is movable or not
    *   @param density: particle density
    *   @param vel: particle velocity
    *   @param w: particle rotation velocity
    */

    __host__
    void makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,
        dfloat density = PARTICLE_DENSITY, dfloat3 vel = dfloat3(0, 0, 0), dfloat3 w = dfloat3(0, 0, 0));

    /*
    *   @brief Create a fixed open cylinder in the bases given the diameter and location of the base center
    *   @param diameter: cylinder diameter in dfloat
    *   @param baseOneCenter: coordinates of the first base center
    *   @param baseTwoCenter: coordinates of the second base center
    *   @param pattern: false: in-line pattern, true: staggered pattern
    */
    __host__
    void makeOpenCylinder(dfloat diameter, dfloat3 baseOneCenter, dfloat3 baseTwoCenter, bool pattern);

    /*
    *   @brief Create the particle in the shape of a capsule with given diameter and center and length between semisphere centers
    *   @param diameter: sphere diameter in dfloat
    *   @param point1: location of the center of the 1st hemisphere cap
    *   @param point2: location of the center of the 2nd hemisphere cap
    *   @param move: particle is movable or not
    *   @param density: particle density
    *   @param vel: particle velocity
    *   @param w: particle rotation velocity
    */
    __host__
    void makeCapsule(dfloat diameter, dfloat3 point1, dfloat3 point2, bool move,
        dfloat density = PARTICLE_DENSITY, dfloat3 vel = dfloat3(0, 0, 0), dfloat3 w = dfloat3(0, 0, 0));

    /*
    *   @brief Create the particle in the shape of an elipsoid
    *   @param diameter: semiaxis a,b,c times 2
    *   @param center : elipsoid center
    *   @param vec : axis of rotation of which the particle will be turned that the local (0,0,1) axis will face in global coordinate
    *   @param angleMag : angle of rotation of which the particle will be turned
    *   @param move: particle is movable or not
    *   @param density: particle density
    *   @param vel: particle velocity
    *   @param w: particle rotation velocity
    */
    __host__
    void makeEllipsoid(dfloat3 diameter, dfloat3 center, dfloat3 vec, dfloat angleMag, bool move,dfloat density, dfloat3 vel, dfloat3 w);

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

#endif