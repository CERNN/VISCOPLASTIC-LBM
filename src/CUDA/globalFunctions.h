/*
*   @file globalFunctions.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Global inline functions, utilized by many files
*   @version 0.3.0
*   @date 16/12/2019
*/

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

#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H

#include <builtin_types.h>
#include <curand_kernel.h>
#include <curand.h>

#include "var.h"
#include "IBM/ibmVar.h"
#include "structs/globalStructs.h"

/*
*   @brief Evaluate the population of equilibrium
*   @param rhow: product between density and population's weight
*   @param uc3: three times the scalar product of velocity and the discretized 
*               velocity i (3 * u * ci)
*   @param p1_mmu: 1 minus the scalar product of velocity and itself times 1.5 
*                  (1 - 1.5 * u * u)
*   @return equilibrium population
*/
__device__
dfloat __forceinline__ gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) -> 
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}


/*
*   @brief Evaluate the force term
*   @param wi: population's weight
*   @param fx_mult: term to multiply force in x
*   @param fy_mult: term to multiply force in y
*   @param fz_mult: term to multiply force in z
*   @return force term for population
*/
__device__
dfloat __forceinline__ gpu_force_term(const dfloat wi, const dfloat fx_mult, 
    const dfloat fy_mult, const dfloat fz_mult)
{
    // i: population number
    // a, b: alfa and beto for somatory
    // F_{i} = w_{i}[3*c_{ia}+9*u_{b}*(c_{ia}-d_{ab}/3)]*F_{a}
    return (wi * (FX*fx_mult + FY*fy_mult + FZ*fz_mult));
}


/*
*   @brief Generate normalized random number in gaussian distribution
*   @param seed: seed of rand
*   @param seq: value that must be different for each call (unless the same 
*               distribution is wanted)
*   @param sigma: sigma value interval for gaussian
*   @return normalized random number
*/
__device__
dfloat __forceinline__ randGauss(const unsigned long long int seed, const unsigned long long int seq, const dfloat sigma)
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


/*
*   @brief Evaluate the position of the element of a 3D matrix ([NX][NY][NZ+4]) 
*         in a 1D array
*   @param x: x axis value
*   @param y: y axis value
*   @param z: z axis value
*   @return element index
*/
__host__ __device__
size_t __forceinline__ idxScalar(unsigned int x, unsigned int y, unsigned int z)
{
    return NX * ((size_t)NY*z + y) + x;
}


/*
*   @brief Evaluate the element of the population of a 4D matrix 
*          ([NX][NY][NZ][Q]) in a 1D array
*   @param x: x axis value
*   @param y: y axis value
*   @param z: z axis value
*   @param d: population number
*   @return element index
*/
__host__ __device__
size_t __forceinline__ idxPop(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int d)
{
    return NX*(NY*((size_t)(NZ+1)*d + z) + y) + x;
}


/*
*   @brief Evaluate the distance between a point in 2D and another
*   @param x1: point 1 x value
*   @param y1: point 1 y value
*   @param x2: point 2 x value
*   @param y2: point 2 y value
*   @return euclidian distance between the two points 
*/
__host__ __device__
dfloat __forceinline__ distPoints2D(const dfloat x1, const dfloat y1, const dfloat x2, const dfloat y2)
{   
    return sqrt((float)(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

/**
*   @brief Copy values from src to dst (shape [NZ, NY, NX])
*
*   @param dst: destiny arrays
*   @param src: source arrays
*/
__global__
void copyFromArray(dfloat3SoA dst, dfloat3SoA src);

#endif // !__GLOBAL_FUNCTIONS_H
