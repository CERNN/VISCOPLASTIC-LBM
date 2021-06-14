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

//collision schemes
//#define HARD_SPHERE
#define SOFT_SPHERE //https://doi.org/10.1201/b11103  chapter 5
//#define EXTERNAL_DUCT_BC //necessary if using annularDuctInterpBounceBack or annularDuctInterpBounceBack
//#define INTERNAL_DUCT_BC //necessary if using annularDuctInterpBounceBack
#define trackerCollisionSize 18


/* -------------------------- COLLISION PARAMETERS -------------------------- */
// Soft sphere
#if defined SOFT_SPHERE


constexpr dfloat PP_FRICTION_COEF = 0.0923; // friction coeficient particle particle
constexpr dfloat PW_FRICTION_COEF = 0.0923; // friction coeficient particle wall
constexpr dfloat PP_REST_COEF = 0.98; // restitution coeficient particle particle
constexpr dfloat PW_REST_COEF = 0.98; // restitution coeficient particle wall
//#define REST_COEF_CORRECTION


//material properties
constexpr dfloat PARTICLE_YOUNG_MODULUS = 385.0;
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
// Hard sphere // WARNING: ONLY FOR 2 OR LESS PARTICLES
#if defined HARD_SPHERE
constexpr dfloat FRICTION_COEF = 0.00923; // friction coeficient
constexpr dfloat REST_COEF = 0.98; // restitution coeficient   
#endif
/* -------------------------------------------------------------------------- */


#endif // !__IBM_BC_H