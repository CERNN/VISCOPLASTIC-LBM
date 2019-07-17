#ifndef VAR_D3Q19_H
#define VAR_D3Q19_H
#define _USE_MATH_DEFINES

#include <builtin_types.h> // for device variables
#if defined(unix) || defined(__unix__) || defined(__unix)
typedef uint32_t __int32;
typedef uint16_t __int16;
#endif
typedef double dfloat;

// --------- OUTPUT DEFINES ---------

#define N_SAVE 800              // saves macroscopics every N_SAVE steps
#define N_RESID 800             // calculates residual every N_RESID steps
#define N_MSG 0                 // prints macroscopics every N_MSG steps

#define SAVE_POP 0              // saves last step's population
#define LOAD_POP 0              // loads population in PATH_DATA of iteration LOAD_POP
                                // (0 does not load, LOAD_MACR must be 0)
#define LOAD_MACR 0             // loads macroscopics in PATH_DATA of iteration LOAD_MACR
                                // (0 does not load, LOAD_POP must be 0)

#define ID_SIM "000"            // prefix for data files

#define PATH_DATA "./tests/"    // path to save simulation's data


// --------- SIMULATION DEFINES ---------
#define SCALE (1)
constexpr int INI_STEP = (LOAD_POP ? LOAD_POP : 
        (LOAD_MACR ? LOAD_MACR : 0));   // initial simulation step 
                                        // (LOAD_POP or LOAD_MACR or 0)

constexpr int N_STEPS = 8000;           // maximum number of time steps

#define RESID 1                         // calculate residual or not
constexpr dfloat RESID_MAX = 1e-4;      // simulation maximal residual


constexpr unsigned int N = 64;          // size of the grid
constexpr unsigned int N_X = N;         // size x of the grid
constexpr unsigned int N_Y = N;         // size y of the grid
constexpr unsigned int N_Z = N;         // size z of the grid

constexpr dfloat U_MAX = 0.05;          // max velocity

constexpr dfloat REYNOLDS = 10.0;       // Reynolds number
constexpr dfloat TAU = 0.5 + 3 *        // Relaxation time calculated for
    (((dfloat)N) * U_MAX / REYNOLDS);   // the Reynolds number

constexpr dfloat OMEGA = 1.0 / TAU;     // (tau) ^ -1
constexpr dfloat T_OMEGA = 1 - OMEGA;   // 1 - omega


constexpr dfloat RHO_0 = 1;                     // initial rho
constexpr dfloat RHO_OUT = RHO_0;               // out fluid's density for parallel plates
constexpr dfloat RHO_IN = RHO_OUT + RHO_0 * 12  // in fluid's density for parallel plates
 * U_MAX * (TAU - 0.5) / (N);


//  --------- TYPE AND SCHEME OF BOUNDARY CONDITIONS ---------
#define BC_LID_DRIVEN_CAVITY 1
#define BC_PARALLEL_PLATES 2
#define BC_SQUARE_DUCT 3                    // TO ADD FORCE TERM 
#define BC_TAYLOR_GREEN_VORTEX 4
#define BC_USED (BC_TAYLOR_GREEN_VORTEX)    // boundary condition to use


// --------- GPU DEFINES ---------
const int nThreads_X = 64;      // multiple of 32 for better performance. N_X must be a multiple of it.
const int nThreads_Y = 1;       // leave as 1
const int nThreads_Z = 1;       // leave as 1

const int CURAND_SEED = 0;  // seed for random numbers for CUDA

// options for boundary conditions
__device__ const dfloat ux_bc[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat uy_bc[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat uz_bc[8] = { 0, U_MAX, 0, 0, 0, 0, 0, 0 };
__device__ const dfloat rho_bc[8] = { RHO_0, RHO_IN, RHO_OUT, 1, 1, 1, 1, 1 };


// ----------- LBM DEFINES ----------- 

/*
------ POPULATIONS -------
    [ i]: (cx,cy,cz)
    [ 0]: ( 0, 0, 0)
    [ 1]: ( 1, 0, 0)
    [ 2]: (-1, 0, 0)
    [ 3]: ( 0, 1, 0)
    [ 4]: ( 0,-1, 0)
    [ 5]: ( 0, 0, 1)
    [ 6]: ( 0, 0,-1)
    [ 7]: ( 1, 1, 0)
    [ 8]: (-1,-1, 0)
    [ 9]: ( 1, 0, 1)
    [10]: (-1, 0,-1)
    [11]: ( 0, 1, 1)
    [12]: ( 0,-1,-1)
    [13]: ( 1,-1, 0)
    [14]: (-1, 1, 0)
    [15]: ( 1, 0,-1)
    [16]: (-1, 0, 1)
    [17]: ( 0, 1,-1)
    [18]: ( 0,-1, 1)
--------------------------
*/

constexpr unsigned char Q = 19;         // number of velocities
constexpr dfloat W_0 = 1.0 / 3;         // population 0 weight (0, 0, 0)
constexpr dfloat W_1 = 1.0 / 18;        // adjacent populations (1, 0, 0)
constexpr dfloat W_2 = 1.0 / 36;        // diagonal populations (1, 1, 0)

// velocities weight vector
__device__ const dfloat w[Q] = { W_0,
    W_1, W_1, W_1, W_1, W_1, W_1,
    W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2
};

// populations velocities vector
__device__ const char c_x[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char c_y[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char c_z[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };


// ---------- MEMORY SIZE ---------- 
const size_t mem_size_pop = sizeof(dfloat) * N_X * N_Y * N_Z * Q;
const size_t mem_size_scalar = sizeof(dfloat) * N_X * N_Y * N_Z;
const size_t mem_size_bc_map = sizeof(__int32) * N_X * N_Y * N_Z;


// PARA DIVIDIR CALCULO DO RESIDUO (not used)
#define SIZE_X_RES 4  
#define SIZE_Y_RES 4
#define SIZE_Z_RES 4
const size_t mem_size_res = (SIZE_X_RES * SIZE_Y_RES * SIZE_Z_RES * sizeof(dfloat));


#endif // VAR_D3Q19_H
