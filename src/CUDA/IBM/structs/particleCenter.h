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
    dfloat3 pos_old;    // Old Particle center position
    dfloat3 vel;        // Particle center translation velocity
    dfloat3 vel_old;    // Old particle center translation velocity
    dfloat3 w;          // Particle center rotation velocity
    dfloat3 w_avg;      // Average particle rotation (used by nodes in movement)
    dfloat3 w_old;      // Old particle center rotation velocity
    dfloat3 f;          // Sum of the forces acting on particle
    dfloat3 f_old;      // Old sum of the forces acting on particle
    dfloat3 M;          // Total momentum acting on particle
    dfloat3 M_old;       // Old total momentum acting on particle
    dfloat3 I;          // I innertia moment I.x = Ixx
    dfloat S;           // Total area of the particle
    dfloat radius;      // Sphere radius
    dfloat volume;      // Particle volume
    bool movable;       // If the particle can move

    /* Constructor */
    particleCenter()
    {
        // All dfloat3 variables are initialized as 0
        pos = dfloat3();
        pos_old = dfloat3();
        vel = dfloat3();
        vel_old = dfloat3();
        w = dfloat3();
        w_avg = dfloat3();
        w_old = dfloat3();
        f = dfloat3();
        f_old = dfloat3();
        M = dfloat3();
        M_old = dfloat3();
        I = dfloat3();

        S = 0;
        radius = 0;
        volume = 0;
        movable = false;
    }
} ParticleCenter;

#endif //__PARTICLE_CENTER_H
