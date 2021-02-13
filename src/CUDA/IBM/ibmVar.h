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
#define NUM_PARTICLES 1
// Number of IBM inner iterations
#define IBM_MAX_ITERATION 3
// Particles diameters
#define PARTICLE_DIAMETER (15)
// Mesh scale for IBM, minimum distance between nodes (lower, more nodes in particle)
#define MESH_SCALE 1.0
// Number of iterations of Coulomb algorithm to optimize the nodes positions
#define MESH_COULOMB 0
// Lock particle rotation (UNUSED)
// #define ROTATION_LOCK true
// Assumed boundary thickness for IBM
#define IBM_THICKNESS (1)
// Transfer and save forces along with macroscopics
#define EXPORT_FORCES false
//collision schemes
#define SOFT_SPHERE
//#define HARD_SPHERE //https://doi.org/10.1201/b11103  chapter 5
/* ------------------------------------------------------------------------- */


/* ---------------------------- IBM OPTIMIZATION --------------------------- */
// Optimize Euler nodes updates for IBM (only recommended to test false
// with a ratio of more than 5% between lagrangian and eulerian nodes)
#define IBM_EULER_OPTIMIZATION false
// "Shell thickness" to consider. The Euler nodes are updated every time 
// the particle moves more than IBM_EULER_UPDATE_DIST value and all nodes with 
// less than IBM_EULER_SHELL_THICKNESS+P_DIST distant from the particle are updated.
// For fixed particles these values does not influence.
// The higher the value, more Euler nodes will be updated every step and
// performance may decrease.
// This value is a tradeoff between:
// Calculate euler nodes that must be updated (lower the value or higher the 
//        particle movement, more frequent this update will be)
// vs.
// Number of euler nodes updated (higher value, higher the eulerian nodes)
// The difference between IBM_EULER_SHELL_THICKNESS and IBM_EULER_UPDATE_DIST must
// be low enough so that the particle doesn't move more than that in 
// IBM_EULER_UPDATE_INTERVAL steps
#define IBM_EULER_SHELL_THICKNESS (2.0)
// MUST BE LOWER OR EQUAL TO IBM_EULER_SHELL_THICKNESS, 
// (equal if IBM_EULER_UPDATE_INTERVAL=1)
#define IBM_EULER_UPDATE_DIST (1.0)
// Every interval to check for update of particles. Note that if particle moves
// more than plannes in this interval it may lead to simulations errors. 
// Leave as 1 if you're not interested in this optimization
#define IBM_EULER_UPDATE_INTERVAL (10)

/* ------------------------------------------------------------------------- */

/* ------------------------- TIME AND SAVE DEFINES ------------------------- */
#define IBM_PARTICLES_SAVE (100)               // Save particles info every given steps (0 not report)
#define IBM_DATA_REPORT (10)                   // Report IBM treated data every given steps (0 not report)
 
#define IBM_DATA_STOP true                 // stop condition by IBM treated data
#define IBM_DATA_SAVE true                 // save reported IBM data to file

#define IBM_PARTICLES_NODES_SAVE true      // Saves particles nodes data
/* ------------------------------------------------------------------------- */

/* ------------------------- FORCES AND DENSITIES --------------------------- */
constexpr dfloat PARTICLE_DENSITY = 1.154639;
constexpr dfloat FLUID_DENSITY = 1;

// Gravity accelaration on particle (Lattice units)
constexpr dfloat GX = 0.0;
constexpr dfloat GY = 0.0;
constexpr dfloat GZ = 0.0;
/* -------------------------------------------------------------------------- */

/* -------------------------- COLLISION PARAMETERS -------------------------- */
// Soft sphere
#if defined SOFT_SPHERE
constexpr dfloat ZETA = 1.0; // Distance threshold
constexpr dfloat STIFF_WALL = 1.0;  // Stiffness parameter wall
constexpr dfloat STIFF_SOFT = 1.0;  // Soft stiffness parameter particle
constexpr dfloat STIFF_HARD = 0.1;  // Hard stiffness parameter particle
#endif
// Hard sphere // WARNING: ONLY FOR 2 OR LESS PARTICLES
#if defined HARD_SPHERE
constexpr dfloat FRIC_COEF = 0.001; // friction coeficient
constexpr dfloat REST_COEF = 1.0; // restitution coeficient   
#endif
/* -------------------------------------------------------------------------- */


/* ------------------------------ STENCIL ----------------------------------- */
// Define only one
// #define STENCIL_2 
#define STENCIL_4 // Peskin Stencil

// Stencil distance
#if defined(STENCIL_2)
#define P_DIST 1
#endif

#if defined(STENCIL_4)
#define P_DIST 2
#endif
/* -------------------------------------------------------------------------- */

// Some prints to test IBM
#define IBM_DEBUG false

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
constexpr unsigned int TOTAL_PCOLLISION_IBM_THREADS = (NUM_PARTICLES*(NUM_PARTICLES+1))/2;
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
