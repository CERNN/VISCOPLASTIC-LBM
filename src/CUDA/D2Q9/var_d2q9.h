#ifndef VAR_D2Q9_H
#define VAR_D2Q9_H
#include <builtin_types.h> // for device variables
#include <stdint.h> // for __int32 on Linux

#if defined(unix) || defined(__unix__) || defined(__unix)
typedef uint32_t __int32;
typedef uint16_t __int16;
#endif

typedef double dfloat;              // double or single precision

// --------- OUTPUT DEFINES ---------
#define N_SAVE 0                // saves macroscopics every N_SAVE steps
#define N_MSG (1000)            // prints message every N_MSG steps
#define ID_SIM "000"

//#define PATH_DATA "./../simulations/D2Q9/dump/"         // path to save simulation's data
#define PATH_DATA "./tests/"            // path to save simulation's data
#define EXT ".csv"                      // file to save extension
#define SEP "\t"                        // csv separator


// --------- SIMULATION DEFINES --------- 
constexpr int N_STEPS = 1000*16;        // maximum number of steps/iterations

constexpr unsigned int N = 2048;        // size of the grid
constexpr unsigned int N_X = N;         // size x of the grid
constexpr unsigned int N_Y = N;         // size y of the grid

constexpr dfloat U_MAX = 0.05;          // max velocity

constexpr dfloat REYNOLDS = 100.0;      // Reynolds number
constexpr dfloat TAU = 0.5 + 3 *        // Relaxation time calculated for
(((dfloat)N) * U_MAX / REYNOLDS);       // the Reynolds number
constexpr dfloat OMEGA = 1.0 / TAU;     // (tau) ^ -1
constexpr dfloat T_OMEGA = 1 - OMEGA;   // 1 - omega


constexpr dfloat RHO_0 = 1;                     // initial rho
constexpr dfloat RHO_OUT = RHO_0;               // out fluid's density for parallel plates
constexpr dfloat RHO_IN = RHO_OUT + RHO_0 * 12  // in fluid's density for parallel plates
* U_MAX * (TAU - 0.5) / (N);

#define RESID 1                         // calculate residual or not
#define N_RESID 1000                    // calculate residual every N_RESID steps
constexpr dfloat RESID_MAX = 1e-4;      // simulation maximal residual


//  --------- TYPE AND SCHEME OF BOUNDARY CONDITIONS ---------
#define BC_LID_DRIVEN_CAVITY 1
#define BC_PARALLEL_PLATES 2
#define BC_USED (BC_PARALLEL_PLATES)    // boundary condition to use

// --------- GPU DEFINES ---------
const int nThreads_X = 128;             // multiple of 32 for better performance. N_X must be a multiple of it.
const int nThreads_Y = 1;

// options for boundary conditions
__device__ const dfloat u_x_bc[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat u_y_bc[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat rho_bc[8] = { RHO_0, RHO_IN, RHO_OUT, 1, 1, 1, 1, 1 };


// ----------- LBM DEFINES ----------- 

/*
--------------------------

------ POPULATIONS -------

------  6   2   5   ------

------  3   0   1   ------

------  7   4   8   ------

--------------------------
*/

constexpr unsigned int Q = 9;           // number of velocities
constexpr dfloat W_0 = 4.0 / 9.0;       // population 0 weight
constexpr dfloat W_1 = 1.0 / 9.0;       // adjacent populations (1, 2, 3, 4) weight
constexpr dfloat W_2 = 1.0 / 36.0;      // diagonal populations (5, 6, 7, 8) weight

// velocities weight vector
__device__ const dfloat w[Q] = {W_0, W_1, W_1, W_1, W_1, W_2, W_2, W_2, W_2};

// populations velocities vector
__device__ const char c_x[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const char c_y[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

// ---------- MEMORY SIZE ---------- 
const size_t mem_size_pop = sizeof(dfloat) * N_X * N_Y * Q;
const size_t mem_size_scalar = sizeof(dfloat) * N_X * N_Y;
const size_t mem_size_bc_map = sizeof(__int16) * N_X * N_Y;

#endif // VAR_D2Q9_H
