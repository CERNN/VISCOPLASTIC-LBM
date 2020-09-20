/*
*   @file ibmStruct.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for IBM particle center
*   @version 0.3.0
*   @date 16/09/2020
*/

#ifndef __PARTICLE_CENTER_H
#define __PARTICLE_CENTER_H

#include "../../structs/globalStructs.h"
#include "../ibmVar.h"

/*
* Describe the particle center properties
*/
typedef struct particleCenter {
    dfloat3 pos;        // Particle center position
    dfloat3 theta;      // Particle center rotation
    dfloat3 vel;        // Particle center translation velocity
    dfloat3 vel_old;    // Particle center translation velocity
    dfloat3 w;          // Particle center rotation velocity
    dfloat3 w_old;      // Particle center rotation velocity
    dfloat3 f;          // Sum of the forces acting in the Particle
    dfloat3 M;          // Sum of moments acting the partticle
    dfloat3 I;          // I innertia moement I.x =  Ixx
    dfloat S;           // total area of the particle
    dfloat rho;         // Particle density relative to fluid
    dfloat mass_p;      // Particle mass
    dfloat mass_f;      // Fluid mass
    dfloat radius;
    bool movable;       // If the particle can move

    /* Constructor */
    particleCenter()
    {
        // All dfloat3 variables are initialized as 0

        S = 0;
        rho = 0;
        mass_p = 0;
        mass_f = 0;
        radius = 0;
        movable = false;
    }
} ParticleCenter;

#endif //__PARTICLE_CENTER_H
