/*
*   @file ibmVar.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Configurations for the immersed boundary method
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __IBM_VAR_H
#define __IBM_VAR_H

#include "../var.h"
#include <stdio.h>

/* -------------------------- IBM GENERAL DEFINES --------------------------- */
// Total number of IB particles in the system
#define NUM_PARTICLES 50
// Number of IBM inner iterations
#define IBM_MAX_ITERATION 10
// Particles diameters
#define PARTICLE_DIAMETER 15.0
// Mesh scale for IBM, minimum distance between nodes (lower, more nodes in particle)
#define MESH_SCALE 1.0
// Number of iterations of Coulomb algorithm to optimize the nodes positions
#define MESH_COULOMB 0
// Lock particle rotation
#define ROTATION_LOCK true
// Assumed boundary thickness for IBM
#define IBM_THICKNESS (1) 
/* ------------------------------------------------------------------------- */


/* ------------------------- FORCES AND DENSITIES --------------------------- */
constexpr dfloat PARTICLE_DENSITY = 2.5;
constexpr dfloat FLUID_DENSITY  = 1;

// Gravity accelaration on particle (Lattice units)
constexpr dfloat GX = 0.0;
constexpr dfloat GY = 0.0;
constexpr dfloat GZ = 5e-5; //4.26E-04;
/* -------------------------------------------------------------------------- */

/* -------------------------- COLLISION PARAMETERS -------------------------- */
// Soft sphere
constexpr dfloat ZETA = 1.0; // Distance threshold
constexpr dfloat STIFF_WALL = 1.0;  // Stiffness parameter wall
constexpr dfloat STIFF_SOFT = 1.0;  // Soft stiffness parameter particle
constexpr dfloat STIFF_HARD = 0.1;  // Hard stiffness parameter particle
/* -------------------------------------------------------------------------- */


/* ------------------------------ STENCIL ----------------------------------- */
// Define only one
// #define STENCIL_2 
#define STENCIL_4 // Peskin Stencil

// Stencil distance
#if defined STENCIL_2
#define P_DIST 1
#endif

#if defined STENCIL_4
#define P_DIST 2
#endif
/* -------------------------------------------------------------------------- */

/* ------------------------ THREADS AND GRIDS FOR IBM ----------------------- */
// Threads for IBM particles
constexpr unsigned int THREADS_PARTICLES_IBM = NUM_PARTICLES > 64 ? 64 : NUM_PARTICLES;
// Grid for IBM particles
constexpr unsigned int GRID_PARTICLES_IBM = 
    (NUM_PARTICLES % THREADS_PARTICLES_IBM ? 
        (NUM_PARTICLES / THREADS_PARTICLES_IBM + 1)
        : (NUM_PARTICLES / THREADS_PARTICLES_IBM));

// For IBM particles collision, the total of threads must be 
// totalThreads = NUM_PARTICLES*(NUM_PARTICLES+1)/2
constexpr unsigned int TOTAL_PCOLLISION_IBM_THREADS = (NUM_PARTICLES*NUM_PARTICLES+1)/2;
// Threads for IBM particles collision 
constexpr unsigned int THREADS_PCOLLISION_IBM = (TOTAL_PCOLLISION_IBM_THREADS > 64) ? 
    64 : TOTAL_PCOLLISION_IBM_THREADS;
// Grid for IBM particles collision
constexpr unsigned int GRID_PCOLLISION_IBM = 
    (TOTAL_PCOLLISION_IBM_THREADS % THREADS_PCOLLISION_IBM ? 
        (TOTAL_PCOLLISION_IBM_THREADS / THREADS_PCOLLISION_IBM + 1)
        : (TOTAL_PCOLLISION_IBM_THREADS / THREADS_PCOLLISION_IBM));
/* -------------------------------------------------------------------------- */

#endif // !__IBM_VAR_H
