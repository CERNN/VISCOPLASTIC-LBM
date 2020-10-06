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
typedef float dfloat;      // single or double precision
#define D3Q19              // velocity set to use
// Comment to disable IBM
#define IBM
/* ------------------------------------------------------------------------- */


/* ----------------------------- OUTPUT DEFINES ---------------------------- */
#define ID_SIM "999"            // prefix for simulation's files
#define PATH_FILES "ParallelPlatesBB"  // path to save simulation's files
                    // the final path is PATH_FILES/ID_SIM
                    // DO NOT ADD "/" AT THE END OF PATH_FILES
/* ------------------------------------------------------------------------- */


/* ------------------------- TIME CONSTANTS DEFINES ------------------------ */
constexpr int N_STEPS = 1000;          // maximum number of time steps
#define MACR_SAVE 100                  // saves macroscopics every MACR_SAVE steps
#define DATA_REPORT 0                // report every DATA_REPORT steps
 
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

// file names to load
#define STR_POP "pop.bin"
#define STR_POP_AUX "pop_aux.bin"
#define STR_RHO "rho.bin"
#define STR_UX "ux.bin"
#define STR_UY "uy.bin"
#define STR_UZ "uz.bin"
/* ------------------------------------------------------------------------- */


/* --------------------------  SIMULATION DEFINES -------------------------- */
constexpr unsigned int N_GPUS = 1;    // Number of GPUS to use

constexpr unsigned int N = 160;
constexpr unsigned int NX = 100;        // size x of the grid 
                                      // (32 multiple for better performance)
constexpr unsigned int NY = 100;        // size y of the grid
constexpr unsigned int NZ = 160;        // size z of the grid in one GPU
constexpr unsigned int NZ_TOTAL = NZ*N_GPUS;       // size z of the grid


constexpr dfloat U_MAX = 0.05;        // max velocity

constexpr dfloat TAU = 0.7294244564;              // relaxation time

constexpr dfloat OMEGA = 1.0/TAU;        // (tau)^-1
constexpr dfloat T_OMEGA = 1-OMEGA;      // 1-omega, for collision
constexpr dfloat TT_OMEGA = 1-0.5*OMEGA; // 1-0.5*omega, for force term

constexpr dfloat RHO_0 = 1;         // initial rho

constexpr dfloat FX = 0;        // force in x
constexpr dfloat FY = 0;        // force in y
constexpr dfloat FZ = 0;     // force in z (flow direction in most cases)

// values options for boundary conditions
__device__ const dfloat UX_BC[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat UY_BC[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat UZ_BC[8] = { 0, U_MAX/2, -U_MAX/2, 0, 0, 0, 0, 0 };
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

/* --------------------------- AUXILIARY DEFINES --------------------------- */ 
#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

constexpr size_t BYTES_PER_GB = (1<<30);
constexpr size_t BYTES_PER_MB = (1<<20);

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)
/* ------------------------------------------------------------------------- */

/* ------------------------------ MEMORY SIZE ------------------------------ */ 
const size_t NUMBER_LBM_NODES = NX*NY*NZ;
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_NODES * Q;
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
const size_t MEM_SIZE_MAP_BC = sizeof(uint32_t) * NUMBER_LBM_NODES;
const size_t TOTAL_NUMBER_LBM_NODES = NX*NY*NZ_TOTAL;
const size_t TOTAL_MEM_SIZE_POP = sizeof(dfloat) * TOTAL_NUMBER_LBM_NODES * Q;
const size_t TOTAL_MEM_SIZE_SCALAR = sizeof(dfloat) * TOTAL_NUMBER_LBM_NODES;
const size_t TOTAL_MEM_SIZE_MAP_BC = sizeof(uint32_t) * TOTAL_NUMBER_LBM_NODES;
/* ------------------------------------------------------------------------- */

#endif // !__VAR_H