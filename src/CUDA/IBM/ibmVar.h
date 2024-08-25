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
#include "ibmBoundaryCondition.h"
#include <stdio.h>
#include <math.h>


/* -------------------------- IBM GENERAL DEFINES --------------------------- */
// Total number of IB particles in the system
#define NUM_PARTICLES 1
// Number of IBM inner iterations
#define IBM_MAX_ITERATION 1
// Particles diameters
#define PARTICLE_DIAMETER (20)
// Change to location of nodes http://dx.doi.org/10.1016/j.jcp.2012.02.026
#define BREUGEM_PARAMETER (0.0)
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


/* ------------------------------------------------------------------------- */


/* ---------------------------- IBM OPTIMIZATION --------------------------- */

/*                      READ THIS FOR OPTIMIZATION OF IBM 
*   The formula below must be used to define the values of the variables below:
*       MAX_MOVE_PER_STEP*(IBM_PARTICLE_SHELL_THICKNESS) < IBM_PARTICLE_UPDATE_INTERVAL
*-  MAX_MOVE_PER_STEP must considers TRANSLATION AND ROTATION movement at the border nodes
*
*   The frequency of update (freq) is:
*       freq = IBM_PARTICLE_UPDATE_DIST/avg_move_per_step
*
*   In order to minimize the frequency update, you must choose values that respecti
*   the first formula, while minimizing the frequency. 
*   Also, you must keep in mind that the higher IBM_PARTICLE_UPDATE_DIST and
*   IBM_PARTICLE_SHELL_THICKNESS, more memory is required
*
*   If the particle max movement is 0.1, examples of good values are:
*       IBM_PARTICLE_SHELL_THICKNESS=5.1
*       IBM_PARTICLE_UPDATE_DIST=2.0
*       IBM_PARTICLE_UPDATE_INTERVAL=50
*   Note that the values satisfies the first formula (0.1*50 < 5.1). 
*   If you wanted to be more conservative, you could decrease IBM_PARTICLE_UPDATE_INTERVAL
*   or increase IBM_PARTICLE_SHELL_THICKNESS. 
*   For lower frequency of update, you could increase IBM_PARTICLE_UPDATE_DIST.
*   Fow lower use of memory, you could decrease IBM_PARTICLE_UPDATE_DIST 
*   or IBM_PARTICLE_SHELL_THICKNESS.
*/

// Shell thickness to consider for each particle. Particle can move at most
// this value in IBM_PARTICLE_UPDATE_INTERVAL steps. If it moves more, simulation 
// may be wrong
#define IBM_PARTICLE_SHELL_THICKNESS (1.0)
// How much a particle must move to be updated (checking is done with frequency)
// of IBM_PARTICLE_UPDATE_INTERVAL
#define IBM_PARTICLE_UPDATE_DIST (0.0)
// Frequency to check if particle has moved more than IBM_PARTICLE_UPDATE_DIST and update
// its nodes in each GPU. If particle nodes move more than (
// IBM_PARTICLE_SHELL_THICKNESS+IBM_PARTICLE_UPDATE_DIST) in this interval, it may lead
// to simulation errors.
#define IBM_PARTICLE_UPDATE_INTERVAL (0)

/*  EULER OPTIMZATION FOLLOWS SIMILAR RULES OF ABOVE OPTIMIZATION. 
*   BUT IT DOES NOT USE CONSIDERABLY MORE MEMORY */
// Optimize Euler nodes updates for IBM (only recommended to test false
// with a ratio of more than 5% between lagrangian and eulerian nodes)
#define IBM_EULER_OPTIMIZATION false
// "Shell thickness" to consider. The Euler nodes are updated every time 
// the particle moves more than IBM_EULER_UPDATE_DIST and all nodes with 
// less than IBM_EULER_SHELL_THICKNESS+IBM_EULER_UPDATE_DIST+P_DIST 
// distant from the particle are updated.
// For fixed particles these values does not influence.
#define IBM_EULER_SHELL_THICKNESS (2.0)
#define IBM_EULER_UPDATE_DIST (0.0)
// Every interval to check for update of particles. Note that if particle moves
// more than planned in this interval it may lead to simulations errors. 
// Leave as 1 if you're not interested in this optimization
#define IBM_EULER_UPDATE_INTERVAL (0)

//Define the discrization coefiecient for the particle movement: 1 = only current time step
// 0.5 =  half current and half previous,  0 = only previous time step information
#define IBM_MOVEMENT_DISCRETIZATION (0.5)  //TODO: its not the correct name, but for now i cant recall it.

/* ------------------------------------------------------------------------- */

/* ------------------------- TIME AND SAVE DEFINES ------------------------- */

#define IBM_PARTICLES_SAVE (100)               // Save particles info every given steps (0 not report)
#define IBM_DATA_REPORT (100)                   // Report IBM treated data every given steps (0 not report)

 
#define IBM_DATA_STOP false                 // stop condition by IBM treated data
#define IBM_DATA_SAVE false                 // save reported IBM data to file

#define IBM_PARTICLES_NODES_SAVE false      // Saves particles nodes data
/* ------------------------------------------------------------------------- */

/* ------------------------- FORCES AND DENSITIES --------------------------- */
constexpr dfloat PARTICLE_DENSITY = 2.0;
constexpr dfloat FLUID_DENSITY = 1.0;

// Gravity accelaration on particle (Lattice units)
constexpr dfloat GX = 0.0;
constexpr dfloat GY = 0.0;
constexpr dfloat GZ = 0.0; //-1.179430e-03/SCALE/SCALE/SCALE;
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
//#define IBM_DEBUG

#ifdef IBM
// Border size is the number of ghost nodes in one size of z for each GPU. 
// These nodes are used for IBM force/macroscopics update/sync
#define MACR_BORDER_NODES (2+(int)((IBM_EULER_UPDATE_DIST+IBM_PARTICLE_SHELL_THICKNESS)+0.99999999))
#else
#define MACR_BORDER_NODES (0)
#endif

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
