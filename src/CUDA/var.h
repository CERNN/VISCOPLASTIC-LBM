/*
*   @file var.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Configurations for the simulation
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __VAR_H
#define __VAR_H

#include <builtin_types.h>  // for devices variables
#include <stdint.h>         // for uint32_t
#define _USE_MATH_DEFINES


/* ------------------------ GENERAL SIMULATION DEFINES ---------------------- */
#define SINGLE_PRECISION    // SINGLE_PRECISION (float) or DOUBLE_PRECISION (double)
#define D3Q19               // velocity set to use (D3Q19 OR D3Q27)
// Comment to disable IBM. Uncomment to enable IBM
#define IBM
/* -------------------------------------------------------------------------- */

/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
// Uncomment the one to use. Comment all to simulate newtonian fluid
// #define POWERLAW
// #define BINGHAM
/* -------------------------------------------------------------------------- */

#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
#endif
#ifdef DOUBLE_PRECISION
    typedef double dfloat;      // double precision
#endif

/* ----------------------------- OUTPUT DEFINES ---------------------------- */

#define ID_SIM "009"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files
                    // the final path is PATH_FILES/ID_SIM
                    // DO NOT ADD "/" AT THE END OF PATH_FILES
/* ------------------------------------------------------------------------- */


/* ------------------------- TIME CONSTANTS DEFINES ------------------------ */
constexpr unsigned int SCALE = 1;

constexpr int N_STEPS = 300000;          // maximum number of time steps
#define MACR_SAVE (0)                  // saves macroscopics every MACR_SAVE steps
#define DATA_REPORT (false)                // report every DATA_REPORT steps
 
#define DATA_STOP false                 // stop condition by treated data
#define DATA_SAVE false                 // save reported data to file

#define POP_SAVE false                  // saves last step's population
/* ------------------------------------------------------------------------- */


/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0; // initial simulation step (0 default)
#define LOAD_POP false      // loads population from binary file (file names
                            // defined below; LOAD_MACR must be false)
#define LOAD_MACR false     // loads macroscopics from binary file (file names
                            // defined below; LOAD_POP must be false)

#define RANDOM_NUMBERS false // to generate random numbers 
                            // (useful for turbulence)

// File names to load
#define STR_POP "pop.bin"
#define STR_POP_AUX "pop_aux.bin"
#define STR_RHO "./fixedSphere/001/001_rho050000.bin"
#define STR_UX "./fixedSphere/001/001_ux050000.bin"
#define STR_UY "./fixedSphere/001/001_uy050000.bin"
#define STR_UZ "./fixedSphere/001/001_uz050000.bin"
// Files for IBM
#define STR_FX "./fixedSphere/001/001_fx050000.bin"
#define STR_FY "./fixedSphere/001/001_fy050000.bin"
#define STR_FZ "./fixedSphere/001/001_fz050000.bin"
// Files for non newtonian
#define STR_OMEGA "omega.bin"
/* ------------------------------------------------------------------------- */


/* --------------------------  SIMULATION DEFINES -------------------------- */
constexpr unsigned int N_GPUS = 1;    // Number of GPUS to use

constexpr int N = 180*SCALE;
constexpr int NX = 180*SCALE;        // size x of the grid 
                                      // (32 multiple for better performance)
constexpr int NY = 180*SCALE;        // size y of the grid
constexpr int NZ = 900*SCALE;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ*N_GPUS;       // size z of the grid

constexpr dfloat U_MAX = 0;           // max velocity

constexpr dfloat TAU = 0.501763;     // relaxation time
constexpr dfloat OMEGA = 1.0/TAU;        // (tau)^-1

constexpr dfloat RHO_0 = 1;         // initial rho

constexpr dfloat FX = 0;        // force in x
constexpr dfloat FY = 0;        // force in y
constexpr dfloat FZ = 0;        // force in z (flow direction in most cases)

// values options for boundary conditions
__device__ const dfloat UX_BC[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat UY_BC[8] = { 0, U_MAX/2, -U_MAX/2, 0, 0, 0, 0, 0 };
__device__ const dfloat UZ_BC[8] = { 0, U_MAX, -U_MAX, 0, 0, 0, 0, 0 };
__device__ const dfloat RHO_BC[8] = { RHO_0, 1, 1, 1, 1, 1, 1, 1 };

constexpr dfloat RESID_MAX = 1e-5;      // maximal residual
/* ------------------------------------------------------------------------- */


/* ------------------------------ GPU DEFINES ------------------------------ */
const int N_THREADS = (NX%64?((NX%32||(NX<32))?NX:32):64); // NX or 32 or 64 
                                    // multiple of 32 for better performance.
const int CURAND_SEED = 0;          // seed for random numbers for CUDA
constexpr float CURAND_STD_DEV = 0.5; // standard deviation for random numbers 
                                    // in normal distribution
/* ------------------------------------------------------------------------- */

/* -------------------- BOUNDARY CONDITIONS TO COMPILE --------------------- */
#define COMP_ALL_BC false                // Compile all boundary conditions
#define COMP_BOUNCE_BACK true          // Compile bounce back
#define COMP_FREE_SLIP true            // Compile free slip
#define COMP_PRES_ZOU_HE false          // Compile pressure zou-he
#define COMP_VEL_ZOU_HE false           // Compile velocity zou he
#define COMP_VEL_BOUNCE_BACK true      // Compile velocityr bounce back
#define COMP_INTERP_BOUNCE_BACK false   // Compile interpolated bounce back
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */
/* -------------------------- DON'T ALTER BELOW!!! ------------------------- */
/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */

#ifdef D3Q19
#include "velocitySets/D3Q19.h"
#endif // !D3Q19
#ifdef D3Q27
#include "velocitySets/D3Q27.h"
#endif // !D3Q27

// Pow function to use
#ifdef SINGLE_PRECISION
    #define POW_FUNCTION powf 
#else
    #define POW_FUNCTION pow
#endif

/* --------------------------- AUXILIARY DEFINES --------------------------- */ 
#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

constexpr size_t BYTES_PER_GB = (1<<30);
constexpr size_t BYTES_PER_MB = (1<<20);

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)
/* ------------------------------------------------------------------------- */

/* ------------------------------ MEMORY SIZE ------------------------------ */ 
// Values for each GPU
const size_t NUMBER_LBM_NODES = NX*NY*NZ;
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_NODES * Q;
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
const size_t MEM_SIZE_MAP_BC = sizeof(uint32_t) * NUMBER_LBM_NODES;
// Values for all GPUs
const size_t TOTAL_NUMBER_LBM_NODES = NX*NY*NZ_TOTAL;
const size_t TOTAL_MEM_SIZE_POP = sizeof(dfloat) * TOTAL_NUMBER_LBM_NODES * Q;
const size_t TOTAL_MEM_SIZE_SCALAR = sizeof(dfloat) * TOTAL_NUMBER_LBM_NODES;
const size_t TOTAL_MEM_SIZE_MAP_BC = sizeof(uint32_t) * TOTAL_NUMBER_LBM_NODES;
/* ------------------------------------------------------------------------- */


#ifndef myMax
#define myMax(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef myMin
#define myMin(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#endif // !__VAR_H