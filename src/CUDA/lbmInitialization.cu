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

#include "lbmInitialization.h"


__host__
void initializationRandomNumbers(
    float* randomNumbers, int seed)
{
    curandGenerator_t gen;

    // Create pseudo-random number generator
    checkCurandStatus(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set generator seed
    checkCurandStatus(curandSetPseudoRandomGeneratorSeed(gen,
        CURAND_SEED));
    
    // Generate NX*NY*NZ floats on device, using normal distribution
    // with mean=0 and std_dev=NORMAL_STD_DEV
    checkCurandStatus(curandGenerateNormal(gen, randomNumbers, NUMBER_LBM_NODES,
        0, CURAND_STD_DEV));

    checkCurandStatus(curandDestroyGenerator(gen));
}


__global__
void gpuInitialization(
    Populations pop,
    Macroscopics macr,
    float* randomNumbers)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalar(x, y, z+MACR_BORDER_NODES);

    gpuMacrInitValue(&macr, randomNumbers, x, y, z);

    for (int i = 0; i < Q; i++)
    {
        // calculate equilibrium population and initialize populations to equilibrium
        dfloat feq = gpu_f_eq(w[i] * macr.rho[index],
            3 * (macr.u.x[index] * cx[i] + macr.u.y[index] * cy[i] + macr.u.z[index] * cz[i]),
            1 - 1.5*(  macr.u.x[index] * macr.u.x[index] 
                 + macr.u.y[index] * macr.u.y[index] 
                 + macr.u.z[index] * macr.u.z[index]));
        
        pop.pop[idxPop(x, y, z, i)] = feq;
        pop.popAux[idxPop(x, y, z, i)] = feq;
    }
}


__device__
void gpuMacrInitValue(
    Macroscopics* macr,
    float* randomNumbers,
    int x, int y, int z)
{
    // +MACR_BORDER_NODES because of the ghost nodes
    macr->rho[idxScalar(x, y, z+MACR_BORDER_NODES)] = RHO_0;
    macr->u.x[idxScalar(x, y, z+MACR_BORDER_NODES)] = 0;
    macr->u.y[idxScalar(x, y, z+MACR_BORDER_NODES)] = 0;
    macr->u.z[idxScalar(x, y, z+MACR_BORDER_NODES)] = 0;

    #ifdef IBM
    macr->f.x[idxScalar(x, y, z+MACR_BORDER_NODES)] = FX;
    macr->f.y[idxScalar(x, y, z+MACR_BORDER_NODES)] = FY;
    macr->f.z[idxScalar(x, y, z+MACR_BORDER_NODES)] = FZ;
    #endif
    #ifdef NON_NEWTONIAN_FLUID
    macr->omega[idxScalar(x, y, z)] = 0;
    #endif

    // Example of usage of random numbers for turbulence in parallel plates flow in z

    /*
    dfloat y_visc = 6.59, ub_f = 15.6, uc_f = 18.2;
​
    // logaritimic velocity profile
    dfloat uz_log, pos = (y < NY/2 ? y + 0.5 : NY - (y + 0.5));
    uz_log = (uc_f*U_TAU)*(pos/del)*(pos/del);
​
    macr->u.z[idxScalar(x, y, z)] = uz_log;
    macr->u.x[idxScalar(x, y, z)] = 0.0;
    macr->u.y[idxScalar(x, y, z)] = 0.0;
    macr->rho[idxScalar(x, y, z)] = RHO_0;
​
    // perturbation
    dfloat pert = 0.1;
    int l = idxScalar(x, y, z), Nt = NUMBER_LBM_NODES;
    macr->u.z[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NZ - Nt*((l + NZ) / Nt)];
    macr->u.x[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NX - Nt*((l + NX) / Nt)];
    macr->u.y[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NY - Nt*((l + NY) / Nt)];
    */
}
