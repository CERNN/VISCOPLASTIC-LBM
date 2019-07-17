/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#include "lbm_d2q9.cuh"

__host__ 
void initialisation(dfloat* f1, dfloat* f2, dfloat* rho, dfloat* u_x, dfloat* u_y)
{
    // blocks in grid
    dim3  grid(N_X / nThreads_X, N_Y / nThreads_Y, 1);
    // threads in block
    dim3  threads(nThreads_X, nThreads_Y, 1);
    gpu_initialisation <<<grid, threads>>> (f1, f2, rho, u_x, u_y);
    getLastCudaError("initialisation error");
}


__global__
void gpu_initialisation(dfloat * __restrict__ f, dfloat * __restrict__ f_post, 
    dfloat * __restrict__ rho, dfloat * __restrict__ u_x, dfloat * __restrict__ u_y)
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 0; i < Q; i++)
    {
        f[index_pop(x, y, i)] = RHO_0 * w[i];
        f_post[index_pop(x, y, i)] = RHO_0 * w[i];
    }

    size_t index = index_scalar(x, y);

    u_x[index] = 0;
    u_y[index] = 0;
    rho[index] = RHO_0;
}


__host__ 
void bc_macr_collision_streaming(dfloat* f1, dfloat* f2, dfloat* rho, dfloat* u_x, dfloat* u_y, NodeTypeMap* ntm, bool save)
{
    // blocks in grid
    dim3  grid(N_X / nThreads_X, N_Y / nThreads_Y, 1);
    // threads in block
    dim3  threads(nThreads_X, nThreads_Y, 1);
    gpu_bc_macr_collision_streaming <<<grid, threads>>>(f1, f2, rho, u_x, u_y, ntm, save);
    getLastCudaError("macr-col-stream error");
}


__global__ 
//void __launch_bounds__(nThreads) 
void gpu_bc_macr_collision_streaming(dfloat * f1, dfloat * __restrict__ f2,
    dfloat * __restrict__ rho, dfloat * __restrict__ u_x, dfloat * __restrict__ u_y, NodeTypeMap* ntm, bool save)
{
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;

    gpu_boundary_conditions(&ntm[index_scalar(x, y)], f1, x, y);

    // adjacent coordinates
    const unsigned short int xp1 = (x + 1) % N_X;
    const unsigned short int yp1 = (y + 1) % N_Y;
    const unsigned short int xm1 = (N_X + x - 1) % N_X;
    const unsigned short int ym1 = (N_Y + y - 1) % N_Y;

    // load populations
    const dfloat f0_var = f1[index_pop(x, y, 0)];
    const dfloat f1_var = f1[index_pop(x, y, 1)];
    const dfloat f2_var = f1[index_pop(x, y, 2)];
    const dfloat f3_var = f1[index_pop(x, y, 3)];
    const dfloat f4_var = f1[index_pop(x, y, 4)];
    const dfloat f5_var = f1[index_pop(x, y, 5)];
    const dfloat f6_var = f1[index_pop(x, y, 6)];
    const dfloat f7_var = f1[index_pop(x, y, 7)];
    const dfloat f8_var = f1[index_pop(x, y, 8)];

    // calc for macroscopics
    // rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    // ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) / rho
    // uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) / rho
    const dfloat rho_var = f0_var + f1_var + f2_var + f3_var + f4_var
        + f5_var + f6_var + f7_var + f8_var;
    const dfloat u_x_var = ((f1_var + f5_var + f8_var) - (f3_var + f6_var + f7_var)) / rho_var;
    const dfloat u_y_var = ((f2_var + f5_var + f6_var) - (f4_var + f7_var + f8_var)) / rho_var;

    if (save)
    {
        rho[index_scalar(x, y)] = rho_var;
        u_x[index_scalar(x, y)] = u_x_var;
        u_y[index_scalar(x, y)] = u_y_var;
    }

    // calc for temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (u_x_var * u_x_var + u_y_var * u_y_var);
    const dfloat rho_w1 = rho_var * W_1;
    const dfloat rho_w2 = rho_var * W_2;
    const dfloat ux3 = 3 * u_x_var;
    const dfloat uy3 = 3 * u_y_var;

    // calc for f_post and streaming to adacent nodes
    // f2 = (1 - 1 / TAU) * f1 + (1 / TAU) * f_eq ->
    // f2 = (1 - OMEGA) * f1 + OMEGA * f_eq ->
    // f2 = T_OMEGA * f1 + OMEGA * f_eq
    // the populations that shoudn't be streamed will be changed by the boundary conditions
    f2[index_pop(x  , y  , 0)] = T_OMEGA * f0_var + OMEGA * gpu_f_eq(rho_var * W_0, 0, p1_muu15);
    f2[index_pop(xp1, y  , 1)] = T_OMEGA * f1_var + OMEGA * gpu_f_eq(rho_w1, ux3, p1_muu15);
    f2[index_pop(x  , yp1, 2)] = T_OMEGA * f2_var + OMEGA * gpu_f_eq(rho_w1, uy3, p1_muu15);
    f2[index_pop(xm1, y  , 3)] = T_OMEGA * f3_var + OMEGA * gpu_f_eq(rho_w1, -ux3, p1_muu15);
    f2[index_pop(x  , ym1, 4)] = T_OMEGA * f4_var + OMEGA * gpu_f_eq(rho_w1, -uy3, p1_muu15);
    f2[index_pop(xp1, yp1, 5)] = T_OMEGA * f5_var + OMEGA * gpu_f_eq(rho_w2, ux3 + uy3, p1_muu15);
    f2[index_pop(xm1, yp1, 6)] = T_OMEGA * f6_var + OMEGA * gpu_f_eq(rho_w2, -ux3 + uy3, p1_muu15);
    f2[index_pop(xm1, ym1, 7)] = T_OMEGA * f7_var + OMEGA * gpu_f_eq(rho_w2, -ux3 - uy3, p1_muu15);
    f2[index_pop(xp1, ym1, 8)] = T_OMEGA * f8_var + OMEGA * gpu_f_eq(rho_w2, ux3 - uy3, p1_muu15);
}


__host__ 
void update_rho_u(dfloat * f, dfloat * rho, dfloat * u_x, dfloat * u_y)
{
    // blocks in grid
    dim3  grid(N_X / nThreads_X, N_Y / nThreads_Y, 1);
    // threads in block
    dim3  threads(nThreads_X, nThreads_Y, 1);

    gpu_update_rho_u <<<grid, threads>>>(f, rho, u_x, u_y);
    getLastCudaError("macroscopics error");
}


__global__
void gpu_update_rho_u(dfloat* __restrict__ f, dfloat* __restrict__ rho, dfloat* __restrict__ u_x, dfloat* __restrict__ u_y)
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= N_X || y >= N_Y)
        return;
    // load populations
    dfloat f0_var = f[index_pop(x, y, 0)];
    dfloat f1_var = f[index_pop(x, y, 1)];
    dfloat f2_var = f[index_pop(x, y, 2)];
    dfloat f3_var = f[index_pop(x, y, 3)];
    dfloat f4_var = f[index_pop(x, y, 4)];
    dfloat f5_var = f[index_pop(x, y, 5)];
    dfloat f6_var = f[index_pop(x, y, 6)];
    dfloat f7_var = f[index_pop(x, y, 7)];
    dfloat f8_var = f[index_pop(x, y, 8)];

    rho[index_scalar(x, y)] = f0_var + f1_var + f2_var + f3_var + f4_var
        + f5_var + f6_var + f7_var + f8_var;
    u_x[index_scalar(x, y)] = ((f1_var + f5_var + f8_var) - (f3_var + f6_var + f7_var)) / rho[index_scalar(x, y)];
    u_y[index_scalar(x, y)] = ((f2_var + f5_var + f6_var) - (f4_var + f7_var + f8_var)) / rho[index_scalar(x, y)];
}


__host__ 
dfloat residual(dfloat* u_x, dfloat* u_y, dfloat* u_x_res, dfloat* u_y_res)
{
    dfloat den = 0.0, num = 0.0;

    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            const dfloat diff_ux = u_x[index_scalar(x, y)] - u_x_res[index_scalar(x, y)];
            const dfloat diff_uy = u_y[index_scalar(x, y)] - u_y_res[index_scalar(x, y)];

            num += std::sqrt(diff_ux * diff_ux + diff_uy * diff_uy);
            den += std::sqrt(u_x[index_scalar(x, y)] * u_x[index_scalar(x, y)] + u_y[index_scalar(x, y)] * u_y[index_scalar(x, y)]);
        }
    if (den != 0)
        return (num / den);
    else
        return 1.0;
}


__host__ 
void equalize_vel(dfloat* u_x, dfloat* u_y, dfloat* u_x_0, dfloat* u_y_0)
{
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            u_x_0[index_scalar(x, y)] = u_x[index_scalar(x, y)];
            u_y_0[index_scalar(x, y)] = u_y[index_scalar(x, y)];
        }
}