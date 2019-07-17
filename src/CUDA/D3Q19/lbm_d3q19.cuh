#pragma once
#ifndef LBM_D3Q19_H
#define LBM_D3Q19_H

#include <cmath>            // for sqrt on residual
#include <math.h>
#include <iostream>
#include <string>           // for reading files names
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <device_launch_parameters.h>

#include "./../common/error_def.cuh"
#include "var_d3q19.h"
#include "boundary_conditions_d3q19.cuh"

/*
*   Initializes populations with values in binary file and 
*   macroscopics given population
*   \param f1_gpu[(N_X, N_Y, N_Z, Q)]: grid of populations to be initialized
*   \param file_pop: file with population's content
*   \param f2_gpu[(N_X, N_Y, N_Z, Q)]: auxiliary grid of populations to be initialized
*   \param rho_gpu[(N_X, N_Y, N_Z)]: nodes' density to initialize
*   \param ux_gpu[(N_X, N_Y, N_Z)]: nodes' x velocity to initialize
*   \param uy_gpu[(N_X, N_Y, N_Z)]: nodes' y velocity to initialize
*   \param uz_gpu[(N_X, N_Y, N_Z)]: nodes' z velocity to initialize
*/
__host__
void pop_initialisation(
    dfloat* f1_gpu, FILE* file_pop,
    dfloat* f2_gpu,
    dfloat* rho_gpu,
    dfloat* ux_gpu,
    dfloat* uy_gpu,
    dfloat* uz_gpu
);

/*
*   Initializes macroscopics with values in binary file and 
*   populations in equilibrium
*   \param f1_gpu[(N_X, N_Y, N_Z, Q)]: grid of populations to be initialized
*   \param f2_gpu[(N_X, N_Y, N_Z, Q)]: auxiliary grid of populations to be initialized
*   \param rho_gpu[(N_X, N_Y, N_Z)]: nodes' density to initialize
*   \param file_rho: file with density content
*   \param ux_gpu[(N_X, N_Y, N_Z)]: nodes' x velocity to initialize
*   \param file_ux: file with ux content
*   \param uy_gpu[(N_X, N_Y, N_Z)]: nodes' y velocity to initialize
*   \param file_uy: file with ux content
*   \param uz_gpu[(N_X, N_Y, N_Z)]: nodes' z velocity to initialize
*   \param file_uz: file with ux content
*/
__host__
void macr_initialisation(
    dfloat* f1_gpu,
    dfloat* f2_gpu,
    dfloat* rho_gpu, FILE* file_rho,
    dfloat* ux_gpu, FILE* file_ux,
    dfloat* uy_gpu, FILE* file_uy,
    dfloat* uz_gpu, FILE* file_uz
);


/*
*   Initializes populations with wquilibrium population (rho=RHO0 and vel=0 by default 
*   or Taylor green vortex initialization)
*   \param f1[(N_X, N_Y, N_Z, Q)]: grid of populations to be initialized
*   \param f2[(N_X, N_Y, N_Z, Q)]: auxiliary grid of populations to be initialized
*   \param rho[(N_X, N_Y, N_Z)]: nodes' density to initialize
*   \param ux[(N_X, N_Y, N_Z)]: nodes' x velocity to initialize
*   \param uy[(N_X, N_Y, N_Z)]: nodes' y velocity to initialize
*   \param uz[(N_X, N_Y, N_Z)]: nodes' z velocity to initialize
*   \param is_macr_init: macroscopics are already initialized or not
*   \param is_taylor_green: is taylor green case
*/
__host__ 
void initialisation(
    dfloat* f1, 
    dfloat* f2, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz,
    bool is_macr_init,
    bool is_taylor_green);
__global__ 
void gpu_initialisation(
    dfloat* f, 
    dfloat* f_post, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz,
    bool is_macr_init);
__global__ 
void gpu_taylor_green_vortex_initialisation(
    dfloat* f, 
    dfloat* f_post, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz);

    
/*
*   Applies boundary conditions, updates macroscopics and then performs collision and streaming
*   \param f1[(N_X, N_Y, N_Z, Q)]: grid of populations to apply BC, perform collision and stream from
*   \param f2[(N_X, N_Y, N_Z, Q)]: grid of populations to stream to
*   \param rho[(N_X, N_Y, N_Z)]: nodes' density values
*   \param ux[(N_X, N_Y, N_Z)]: nodes' x velocity values
*   \param uy[(N_X, N_Y, N_Z)]: nodes' y velocity values
*   \param uz[(N_X, N_Y, N_Z)]: nodes' z velocity values
*   \param ntm[(N_X, N_Y, N_Z)]: boundary conditions map
*   \param save: save macroscopics
*   \param iter: iteration
*   \param stream: stream for kernel launch
*/
__host__ 
void bc_macr_collision_streaming(
    dfloat* f1, 
    dfloat* f2, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz, 
    NodeTypeMap* ntm, 
    bool save,
    int iter,
    cudaStream_t* stream);

__global__ 
void gpu_bc_macr_collision_streaming(
    dfloat * f1, 
    dfloat * __restrict__ f2,
    dfloat * __restrict__ rho, 
    dfloat * __restrict__ ux, 
    dfloat * __restrict__ uy, 
    dfloat* __restrict__ uz, 
    NodeTypeMap* ntm, 
    bool save,
    int iter);


/*
*   Update densisty and velocity of all nodes
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param rho[(N_X, N_Y, N_Z)]: nodes' density values to be updated
*   \param u_x[(N_X, N_Y, N_Z)]: nodes' x velocity values to be updated
*   \param u_y[(N_X, N_Y, N_Z)]: nodes' y velocity values to be updated
*   \param u_z[(N_X, N_Y, N_Z)]: nodes' z velocity values to be updated
*/
__host__ 
void update_rho_u(
    dfloat* f, 
    dfloat* rho, 
    dfloat* u_x, 
    dfloat* u_y, 
    dfloat* u_z);

__global__ 
void gpu_update_rho_u(
    dfloat* f, 
    dfloat* rho, 
    dfloat* __restrict__ u_x, 
    dfloat* __restrict__ u_y, 
    dfloat* __restrict__ u_z);


/*
*   Calculate the residual between two populations
*   \param u_y[(N_X, N_Y, N_Z)]: nodes' current x velocity values
*   \param u_y[(N_X, N_Y, N_Z)]: nodes' current y velocity values
*   \param u_z[(N_X, N_Y, N_Z)]: nodes' current z velocity values
*   \param u_x_0[(N_X, N_Y, N_Z)]: nodes' reference x velocity values
*   \param u_y_0[(N_X, N_Y, N_Z)]: nodes' reference y velocity values
*   \param u_z_0[(N_X, N_Y, N_Z)]: nodes' reference z velocity values
*   \return residual value
*/
__host__ 
dfloat residual(
    dfloat* u_x, 
    dfloat* u_y, 
    dfloat* u_z, 
    dfloat* u_x_res, 
    dfloat* u_y_res, 
    dfloat* u_z_res);
__host__
dfloat residual_parallel(
    dfloat* u_x, 
    dfloat* u_y, 
    dfloat* u_z, dfloat* 
    u_x_res, 
    dfloat* u_y_res, 
    dfloat* u_z_res, 
    dfloat* num_gpu, 
    dfloat* den_gpu);
__global__
void gpu_residual(
    dfloat* u_x,
    dfloat* u_y,
    dfloat* u_z,
    dfloat* u_x_res,
    dfloat* u_y_res,
    dfloat* u_z_res, 
    dfloat* num, 
    dfloat* den
);


/*
*   Equalizes velocities (u_x_0 = u_x and u_y_0 = u_y and u_z_0 = u_z)
*   \param u_x[(N_X, N_Y, N_Z)]: nodes' x velocity values (reference)
*   \param u_y[(N_X, N_Y, N_Z)]: nodes' y velocity values (reference)
*   \param u_x_0[(N_X, N_Y, N_Z)]: nodes' x velocity to equalize
*   \param u_y_0[(N_X, N_Y, N_Z)]: nodes' y velocity to equalize
*/
__host__ 
void equalize_vel(
    dfloat* u_x, 
    dfloat* u_y, 
    dfloat* u_z, 
    dfloat* u_x_0, 
    dfloat* u_y_0, 
    dfloat* u_z_0);

#endif // LBM_D3Q19_H
