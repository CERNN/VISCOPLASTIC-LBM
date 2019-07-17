#ifndef LBM_D2Q9_H
#define LBM_D2Q9_H

#include <cmath>            // for sqrt on residual
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <device_launch_parameters.h>

#include "./../common/error_def.cuh"
#include "boundary_conditions_d2q9.cuh"
#include "var_d2q9.h"


/*
*   Initializes populations
*   \param f1[(N_X, N_Y, Q)]: grid of populations from 0 to 8 to be initialized
*   \param f2[(N_X, N_Y, Q)]: auxiliary grid of populations from 0 to 8 to be initialized
*   \param rho[(N_X, N_Y)]: nodes' density to initialize
*   \param u_x[(N_X, N_Y)]: nodes' x velocity to initialize
*   \param u_y[(N_X, N_Y)]: nodes' y velocity to initialize
*/
__host__ 
void initialisation(dfloat* f1, dfloat* f2, dfloat* rho, dfloat* u_x, dfloat* u_y);
__global__ 
void gpu_initialisation(dfloat* __restrict__ f, dfloat* __restrict__ f_post, dfloat* __restrict__ rho, dfloat* __restrict__ u_x, dfloat* __restrict__ u_y);


/*
*   Applies boundary conditions, updates macroscopics and then performs collision and streaming
*   \param f1[(N_X, N_Y, Q)]: grid of populations from 0 to 8 to perform collision and stream from
*   \param f2[(N_X, N_Y, Q)]: grid of populations from 0 to 8 to stream to
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*   \param save: save macroscopics
*/
__host__ 
void bc_macr_collision_streaming(dfloat* fx, dfloat* f2, dfloat* rho, dfloat* u_x, dfloat* u_y, NodeTypeMap* ntm, bool save);
__global__ 
void gpu_bc_macr_collision_streaming(dfloat * f1, dfloat * __restrict__ f2,
    dfloat * __restrict__ rho, dfloat * __restrict__ u_x, dfloat * __restrict__ u_y, NodeTypeMap* ntm, bool save);


/*
*   Update densisty and velocity of all nodes
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param rho[(N_X, N_Y)]: nodes' density values to be updated
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values to be updated
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be updated
*/
__host__ 
void update_rho_u(dfloat* f, dfloat* rho, dfloat* u_x, dfloat* u_y);
__global__ 
void gpu_update_rho_u(dfloat* __restrict__ f, dfloat* __restrict__ rho, dfloat* __restrict__ u_x, dfloat* __restrict__ u_y);


/*
*   Calculate the residual of two populations
*   \param u_y[(N_X, N_Y)]: nodes' current x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' current y velocity values
*   \param u_x_0[(N_X, N_Y)]: nodes' reference x velocity values
*   \param u_y_0[(N_X, N_Y)]: nodes' reference y velocity values
*   \return residual value
*/
__host__ 
dfloat residual(dfloat* u_x, dfloat* u_y, dfloat* u_x_res, dfloat* u_y_res);


/*
*   Equalizes velocities (u_x_0 = u_x and u_y_0 = u_y)
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values (reference)
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values (reference)
*   \param u_x_0[(N_X, N_Y)]: nodes' x velocity to equalize
*   \param u_y_0[(N_X, N_Y)]: nodes' y velocity to equalize
*/
__host__ 
void equalize_vel(dfloat* u_x, dfloat* u_y, dfloat* u_x_0, dfloat* u_y_0);

#endif // LBM_D2Q9_H
