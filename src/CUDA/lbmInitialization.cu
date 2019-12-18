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
void initializationPop( 
    Populations* pop,
    FILE* filePop)
{
    dfloat* tmp = (dfloat*)malloc(memSizePop);
    if (filePop != NULL)
    {
        fread(tmp, memSizePop, 1, filePop);
        
        for(size_t idx = 0; idx < numberNodes*Q; idx++)
        {
            pop->pop[idx] = tmp[idx];
            pop->popAux[idx] = tmp[idx];
        }
    }
    free(tmp);
}


__host__
void initializationMacr(
    Macroscopics* macr,
    FILE* fileRho,
    FILE* fileUx,
    FILE* fileUy,
    FILE* fileUz)
{
    dfloat* tmp = (dfloat*)malloc(memSizeScalar);
    if (fileRho != NULL)
    {
        fread(tmp, memSizeScalar, 1, fileRho);
        for(size_t idx = 0; idx < numberNodes; idx++)
            macr->rho[idx] = tmp[idx];
    }
    if (fileUx != NULL)
    {
        fread(tmp, memSizeScalar, 1, fileUx);
        for(size_t idx = 0; idx < numberNodes; idx++)
            macr->ux[idx] = tmp[idx];
    }
    if (fileUy != NULL)
    {
        fread(tmp, memSizeScalar, 1, fileUy);
        for(size_t idx = 0; idx < numberNodes; idx++)
            macr->uy[idx] = tmp[idx];
    }
    if (fileUz != NULL)
    {
        fread(tmp, memSizeScalar, 1, fileUz);
        for(size_t idx = 0; idx < numberNodes; idx++)
            macr->uz[idx] = tmp[idx];
    }
    free(tmp);
}


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
    checkCurandStatus(curandGenerateNormal(gen, randomNumbers, numberNodes,
        0, CURAND_STD_DEV));

    checkCurandStatus(curandDestroyGenerator(gen));
}


__global__
void gpuInitialization(
    Populations* pop,
    Macroscopics* macr,
    bool isMacrInit,
    float* randomNumbers)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalar(x, y, z);

    if (!isMacrInit)
    {
        gpuMacrInitValue(macr, randomNumbers, x, y, z);
    }

    for (int i = 0; i < Q; i++)
    {
        // calculate equilibrium population and initialize populations to equilibrium
        dfloat feq = gpu_f_eq(w[i] * macr->rho[index],
            3 * (macr->ux[index] * cx[i] + macr->uy[index] * cy[i] + macr->uz[index] * cz[i]),
            1 - 1.5*(  macr->ux[index] * macr->ux[index] 
                 + macr->uy[index] * macr->uy[index] 
                 + macr->uz[index] * macr->uz[index]));
        
        pop->pop[idxPop(x, y, z, i)] = feq;
        pop->popAux[idxPop(x, y, z, i)] = feq;
    }
}


__device__
void gpuMacrInitValue(
    Macroscopics* macr,
    float* randomNumbers,
    int x, int y, int z)
{
    macr->rho[idxScalar(x, y, z)] = RHO_0;
    macr->ux[idxScalar(x, y, z)] = 0;
    macr->uy[idxScalar(x, y, z)] = 0;
    macr->uz[idxScalar(x, y, z)] = 0;

    // Example of usage of random numbers for turbulence in parallel plates flow in z

    /*
    dfloat y_visc = 6.59, ub_f = 15.6, uc_f = 18.2;
​
    // logaritimic velocity profile
    dfloat uz_log, pos = (y < NY/2 ? y + 0.5 : NY - (y + 0.5));
    uz_log = (uc_f*U_TAU)*(pos/del)*(pos/del);
​
    macr->uz[idxScalar(x, y, z)] = uz_log;
    macr->ux[idxScalar(x, y, z)] = 0.0;
    macr->uy[idxScalar(x, y, z)] = 0.0;
    macr->rho[idxScalar(x, y, z)] = RHO_0;
​
    // perturbation
    dfloat pert = 0.1;
    int l = idxScalar(x, y, z), Nt = numberNodes;
    macr->uz[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NZ - Nt*((l + NZ) / Nt)];
    macr->ux[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NX - Nt*((l + NX) / Nt)];
    macr->uy[idxScalar(x, y, z)] += (ub_f*U_TAU)*pert*randomNumbers[l + NY - Nt*((l + NY) / Nt)];
    */
}
