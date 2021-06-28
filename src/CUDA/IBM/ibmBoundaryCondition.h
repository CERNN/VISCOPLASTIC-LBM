/*
*   @file ibmVar.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @brief Configurations for the boundary conditions that affect the immersed boundary method
*   @version 0.3.0
*   @date 14/06/2021
*/

#ifndef __IBM_BC_H
#define __IBM_BC_H

#include "../var.h"
#include "ibmVar.h"
#include <stdio.h>
#include <math.h>

/* -------------------------- BOUNDARY CONDITIONS -------------------------- */

// --- X direction ---
//#define IBM_BC_X_WALL
#define IBM_BC_X_PERIODIC

#ifdef IBM_BC_X_WALL
    // TODO: not implemented yet
    #define IBM_BC_X_WALL_UY 0.0
    #define IBM_BC_X_WALL_UZ 0.0
#endif //IBM_BC_X_WALL

#ifdef IBM_BC_X_PERIODIC
    #define IBM_BC_X_0 (0)
    #define IBM_BC_X_E (NX-0)
#endif //IBM_BC_X_PERIODIC



// --- Y direction ---
#define IBM_BC_Y_WALL
//#define IBM_BC_Y_PERIODIC

#ifdef IBM_BC_Y_WALL
    // TODO: not implemented yet
    #define IBM_BC_Y_WALL_UX 0.0
    #define IBM_BC_Y_WALL_UZ 0.0
#endif //IBM_BC_Y_WALL

#ifdef IBM_BC_Y_PERIODIC
    #define IBM_BC_Y_0 0
    #define IBM_BC_Y_E (NY)
#endif //IBM_BC_Y_PERIODIC



// --- Z direction ---
#define IBM_BC_Z_WALL
//#define IBM_BC_Z_PERIODIC

#ifdef IBM_BC_Z_WALL
    // TODO: not implemented yet
    #define IBM_BC_Z_WALL_UX 0.0
    #define IBM_BC_Z_WALL_UY 0.0
#endif //IBM_BC_Z_WALL

#ifdef IBM_BC_Z_PERIODIC
    #define IBM_BC_Z_0 0
    #define IBM_BC_Z_E (NZ_TOTAL)
#endif //IBM_BC_Z_PERIODIC


// ----------- DUCT BOUNDARY CONDITION ----------- 

//#define EXTERNAL_DUCT_BC //necessary if using annularDuctInterpBounceBack or annularDuctInterpBounceBack
//#define INTERNAL_DUCT_BC //necessary if using annularDuctInterpBounceBack

#ifdef EXTERNAL_DUCT_BC // same as in circularDuctInterpBounceBack
    #define EXTERNAL_DUCT_BC_RADIUS (NY/2.0-0.5)
#endif //EXTERNAL_DUCT_BC

#ifdef INTERNAL_DUCT_BC // same as in annularDuctInterpBounceBack
    #define INTERNAL_DUCT_BC_RADIUS (R/4.0)
#endif //INTERNAL_DUCT_BC

/* -------------------------------------------------------------------------- */


/* -------------------------- COLLISION PARAMETERS -------------------------- */

//collision schemes
//#define HARD_SPHERE
#define SOFT_SPHERE //https://doi.org/10.1201/b11103  chapter 5
#define trackerCollisionSize 18


// Soft sphere
#if defined SOFT_SPHERE


constexpr dfloat PP_FRICTION_COEF = 0.0923; // friction coeficient particle particle
constexpr dfloat PW_FRICTION_COEF = 0.0923; // friction coeficient particle wall
constexpr dfloat PP_REST_COEF = 0.98; // restitution coeficient particle particle
constexpr dfloat PW_REST_COEF = 0.98; // restitution coeficient particle wall
//#define REST_COEF_CORRECTION


//material properties
constexpr dfloat PARTICLE_YOUNG_MODULUS = 10.0;
constexpr dfloat PARTICLE_POISSON_RATIO = 0.24;
constexpr dfloat PARTICLE_SHEAR_MODULUS = PARTICLE_YOUNG_MODULUS / (2.0+2.0*PARTICLE_POISSON_RATIO);

constexpr dfloat WALL_YOUNG_MODULUS = 385.0;
constexpr dfloat WALL_POISSON_RATIO = 0.24;
constexpr dfloat WALL_SHEAR_MODULUS = WALL_YOUNG_MODULUS / (2.0+2.0*WALL_POISSON_RATIO);


//Hertzian contact theory -  Johnson 1985
constexpr dfloat PP_STIFFNESS_NORMAL_CONST = (4.0/3.0) / ((1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS + (1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS);
constexpr dfloat PW_STIFFNESS_NORMAL_CONST = (4.0/3.0) / ((1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS + (1-WALL_POISSON_RATIO*WALL_POISSON_RATIO)/WALL_YOUNG_MODULUS);
//Mindlin theory 1949
constexpr dfloat PP_STIFFNESS_TANGENTIAL_CONST =  4.0 * SQRT_2 / ((2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS + (2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS);
constexpr dfloat PW_STIFFNESS_TANGENTIAL_CONST =  4.0 * SQRT_2 / ((2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS + (2-WALL_POISSON_RATIO)/WALL_SHEAR_MODULUS);

//#define LUBRICATION_FORCE
#if defined LUBRICATION_FORCE
    constexpr dfloat MAX_LUBRICATION_DISTANCE = 2;
    constexpr dfloat MIN_LUBRICATION_DISTANCE = 0.001;
#endif


#endif
/* -------------------------------------------------------------------------- */


#endif // !__IBM_BC_H