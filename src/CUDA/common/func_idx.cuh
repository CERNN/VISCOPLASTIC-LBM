#ifndef FUNC_IDX_CUH
#define FUNC_IDX_CUH

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

// TODO: CHANGE FILE NAME

#define D3Q19   // for set of velocities to use

#include <builtin_types.h>
#include <curand_kernel.h>
#include <curand.h>

#ifdef D2Q9
    #include "./../D2Q9/var_d2q9.h"
#endif

#ifdef D3Q19
    #include "./../D3Q19/var_d3q19.h"
#endif

/*
*   Evaluate the population of equilibrium
*   \param rhow: product between density and population's weight
*   \param uc3: three times the scalar product of velocity and the discretized velocity i (3 * u * ci)
*   \param p1_mmu: 1 minus the scalar product of velocity and itself times 1.5 (1 - 1.5 * u * u)
*   \return equilibrium population
*/
__device__
dfloat __forceinline__ gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) -> 
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}


/*
*   Generate normalized random number in gaussian distribution, given sigma
*   \param seed: seed of rand
*   \param seq: value that must be different for each call (unless the same distribution is wanted)
*   \param sigma: sigma value interval for gaussian
*   \return normalized random number
*/
__device__
dfloat __forceinline__ rand_gauss(const unsigned long long int seed, const unsigned long long int seq, const dfloat sigma)
{
    curandState_t state;

    /* initialize the state */
    curand_init(seed, /* the seed controls the sequence of random values that are produced */
        seq, /* the sequence number is important for multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);
    dfloat rand = curand_normal_double(&state);
    while(rand > sigma || rand < -sigma)
        rand = curand_normal_double(&state);
    return rand/sigma;
}


#ifdef D2Q9
/*
*   Evaluate the position of the element of a 2D matrix ([N_Y][N_X]) in a 1D array
*   \param x: value in first dimension
*   \param y: value in second dimension
*   \return element index
*/
__host__ __device__
size_t __forceinline__ index_scalar(unsigned int x, unsigned int y)
{
    return N_X*y + x;
}


/*
*   Evaluate the position of the population of a 3D matrix ([N_X][N_Y][Q]) in a 1D array
*   \param x: x axis value
*   \param y: y axis value
*   \param d: population number
*   \return element index
*/
__host__ __device__
size_t __forceinline__ index_pop(const unsigned int x, const unsigned int y, const unsigned int d)
{
    return N_X*(N_Y*d + y) + x;
}


/*
*   Evaluate the position of the population of a node in a 1D array
*   \param index_pop_0: index of node's population 0
*   \param d: population's number
*   \return element index
*/
__host__ __device__
size_t __forceinline__ index_pop_node(const size_t index_pop_0, const unsigned int d)
{
    return (index_pop_0 + d * N_X * N_Y);
}

#endif // D2Q9

#ifdef D3Q19
/*
*   Evaluate the position of the element of a 3D matrix ([N_X][N_Y][N_Z]) in a 1D array
*   \param x: x axis value
*   \param y: y axis value
*   \param z: z axis value
*   \return element index
*/
__host__ __device__
size_t __forceinline__ index_scalar_d3(unsigned int x, unsigned int y, unsigned int z)
{
    return N_X * (N_Y*z + y) + x;
}


/*
*   Evaluate the position of the population of a 4D matrix ([N_X][N_Y][N_Z][Q]) in a 1D array
*   \param x: x axis value
*   \param y: y axis value
*   \param z: z axis value
*   \param d: population number
*   \return population index
*/
__host__ __device__
size_t __forceinline__ index_pop_d3q19(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int d)
{   return N_X*(N_Y*(N_Z*d + z) + y) + x;

}


/*
*   Evaluate the position of the population in a thread block in a 1D array
*   \param tx: thread x axis value
*   \param ty: thread y axis value
*   \param tz: thread z axis value
*   \param d: population number
*   \return population index
*/
__host__ __device__
short int __forceinline__ index_pop_in_block_d3q19(const unsigned short int tx, const unsigned short int ty, const unsigned short int tz, const unsigned short int d)
{   
    return nThreads_X*(nThreads_Y*(nThreads_Z*d + tz) + ty) + tx;
}

/*
*   Evaluates de macroscopics of a node in D3Q19
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*   \param x: node's x value
*   \param z: node's y value
*   \param z: node's y value
*   \param rho: pointer to rho value
*   \param u_x: pointer to ux value
*   \param u_y: pointer to uz value
*   \param u_z: pointer to uz value
*/
__device__ __host__
void __inline__ evaluate_macr_d3q19(dfloat* __restrict__ f,
    const unsigned short int x,
    const unsigned short int y,
    const unsigned short int z,
    dfloat* rho,
    dfloat* u_x,
    dfloat* u_y,
    dfloat* u_z)
{
    dfloat f_aux[Q];
    for (unsigned char i = 0; i < Q; i++)
        f_aux[i] = f[index_pop_d3q19(x, y, z, i)];

    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i]) / rho
    // uy = sum(f[i]*cy[i]) / rho
    // uz = sum(f[i]*cz[i]) / rho
    dfloat rho_var = f_aux[0] + f_aux[1] + f_aux[2] + f_aux[3] + f_aux[4] + f_aux[5] + f_aux[6]
        + f_aux[7] + f_aux[8] + f_aux[9] + f_aux[10] + f_aux[11] + f_aux[12] + f_aux[13] + f_aux[14]
        + f_aux[15] + f_aux[16] + f_aux[17] + f_aux[18];
    dfloat u_x_var = ((f_aux[1] + f_aux[7] + f_aux[9] + f_aux[13] + f_aux[15])
        - (f_aux[2] + f_aux[8] + f_aux[10] + f_aux[14] + f_aux[16])) / (rho_var);
    dfloat u_y_var = ((f_aux[3] + f_aux[7] + f_aux[11] + f_aux[14] + f_aux[17])
        - (f_aux[4] + f_aux[8] + f_aux[12] + f_aux[13] + f_aux[18])) / (rho_var);
    dfloat u_z_var = ((f_aux[5] + f_aux[9] + f_aux[11] + f_aux[16] + f_aux[18])
        - (f_aux[6] + f_aux[10] + f_aux[12] + f_aux[15] + f_aux[17])) / (rho_var);
    if (rho != nullptr)
        *rho = rho_var;
    if (u_x != nullptr)
        *u_x = u_x_var;
    if (u_y != nullptr)
        *u_y = u_y_var;
    if (u_z != nullptr)
        *u_z = u_z_var;
}


/*
* FUNCAO AUXILIAR PARA ESCRITA DO RESIDUO
*/
__host__ __device__
size_t __forceinline__ index_residual_d3(const unsigned short x, const unsigned short y, const unsigned short z)
{
    return (int)(N_X / SIZE_X_RES)*((int)(N_Y / SIZE_Y_RES)*(int)(z / SIZE_Z_RES) + (int)(y / SIZE_Y_RES)) + (int)(x / SIZE_X_RES);
}

#endif // D3Q19

#endif // !FUNC_IDX_CUH
