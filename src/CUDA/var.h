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

#define ID_SIM "001"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files

                    // the final path is PATH_FILES/ID_SIM
                    // DO NOT ADD "/" AT THE END OF PATH_FILES
/* ------------------------------------------------------------------------- */


/* ------------------------- TIME CONSTANTS DEFINES ------------------------ */

constexpr unsigned int SCALE = 1;
constexpr int N_STEPS = 1000;          // maximum number of time steps
#define MACR_SAVE (5)                  // saves macroscopics every MACR_SAVE steps
#define DATA_REPORT (false)                // report every DATA_REPORT steps

 
#define DATA_STOP false                 // stop condition by treated data
#define DATA_SAVE false                 // save reported data to file

// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE 20000
/* ------------------------------------------------------------------------- */


/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0; // initial simulation step (0 default)
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)
#define RANDOM_NUMBERS false    // to generate random numbers 
                                // (useful for turbulence)

// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "teste"
/* ------------------------------------------------------------------------- */



/* --------------------------  SIMULATION DEFINES -------------------------- */
constexpr unsigned int N_GPUS = 1;    // Number of GPUS to use
constexpr unsigned int GPUS_TO_USE[N_GPUS] = {0};    // Which GPUs to use



constexpr int N = 64*SCALE;
constexpr int NX = 64*SCALE;        // size x of the grid 
                                      // (32 multiple for better performance)
constexpr int NY = 64*SCALE;        // size y of the grid
constexpr int NZ = 64*SCALE/N_GPUS;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ*N_GPUS;       // size z of the grid

constexpr dfloat U_MAX = 0;           // max velocity

constexpr dfloat TAU = 0.9;     // relaxation time
constexpr dfloat OMEGA = 1.0/TAU;        // (tau)^-1

constexpr dfloat RHO_0 = 1;         // initial rho

constexpr dfloat FX = 1e-4;        // force in x
constexpr dfloat FY = 0;        // force in y
constexpr dfloat FZ = 1e-4;        // force in z (flow direction in most cases)

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
#define COMP_FREE_SLIP false            // Compile free slip
#define COMP_PRES_ZOU_HE false          // Compile pressure zou-he
#define COMP_VEL_ZOU_HE false           // Compile velocity zou he
#define COMP_VEL_BOUNCE_BACK false      // Compile velocityr bounce back
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
// There are ghosts nodes in z for IBM macroscopics (velocity, density, force)
#define NUMBER_LBM_IB_MACR_NODES (size_t)(NX*NY*(NZ+MACR_BORDER_NODES*2))
// There is 1 ghost node in z for communication multi-gpu
const size_t NUMBER_LBM_POP_NODES = NX*NY*(NZ+1);
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_POP_NODES * Q;
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
#define MEM_SIZE_IBM_SCALAR (size_t)(sizeof(dfloat) * NUMBER_LBM_IB_MACR_NODES)
const size_t MEM_SIZE_MAP_BC = sizeof(uint32_t) * NUMBER_LBM_NODES;
// Values for all GPUs
const size_t TOTAL_NUMBER_LBM_NODES = NX*NY*NZ_TOTAL;
#define TOTAL_NUMBER_LBM_IB_MACR_NODES (size_t)(NUMBER_LBM_IB_MACR_NODES * N_GPUS)
const size_t TOTAL_NUMBER_LBM_POP_NODES = NUMBER_LBM_POP_NODES * N_GPUS;
const size_t TOTAL_MEM_SIZE_POP = MEM_SIZE_POP * N_GPUS;
#define TOTAL_MEM_SIZE_IBM_SCALAR (size_t)(MEM_SIZE_IBM_SCALAR * N_GPUS)
const size_t TOTAL_MEM_SIZE_SCALAR = MEM_SIZE_SCALAR * N_GPUS;
const size_t TOTAL_MEM_SIZE_MAP_BC = MEM_SIZE_MAP_BC * N_GPUS;
/* ------------------------------------------------------------------------- */


#ifndef myMax
#define myMax(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef myMin
#define myMin(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#endif // !__VAR_H