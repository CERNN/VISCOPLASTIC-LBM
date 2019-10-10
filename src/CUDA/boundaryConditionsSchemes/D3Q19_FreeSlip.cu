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

#include "D3Q19_FreeSlip.h"


__device__ 
void gpuBCFreeSlipN(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y-1, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y-1, z, 14)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y-1, z, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y-1, z, 7)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y-1, z, 11)];
}


__device__
void gpuBCFreeSlipS(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y+1, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y+1, z, 13)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y+1, z, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y+1, z, 8)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y+1, z, 12)];
}


__device__ 
void gpuBCFreeSlipW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x+1, y, z, 2)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x+1, y, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x+1, y, z, 16)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x+1, y, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x+1, y, z, 10)];
}


__device__ 
void gpuBCFreeSlipE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x-1, y, z, 1)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x-1, y, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x-1, y, z, 15)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x-1, y, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x-1, y, z, 9)];
}


__device__ 
void gpuBCFreeSlipF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z-1, 5)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z-1, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z-1, 18)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z-1, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z-1, 11)];
}


__device__ 
void gpuBCFreeSlipB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z+1, 6)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z+1, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z+1, 17)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z+1, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z+1, 12)];
}


__device__ 
void gpuBCFreeSlipNW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 11)];
    //Dead Pop are: [7, 8]
}


__device__ 
void gpuBCFreeSlipNE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 17)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 11)];
}


__device__ 
void gpuBCFreeSlipNF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 9)];
    //Dead Pop are: [17, 18]
}


__device__ 
void gpuBCFreeSlipNB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [11, 12]
}


__device__ 
void gpuBCFreeSlipSW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 18)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 12)];
    //Dead Pop are: [13, 14]
}


__device__ 
void gpuBCFreeSlipSE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 12)];
    //Dead Pop are: [7, 8]
}


__device__ 
void gpuBCFreeSlipSF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [11, 12]
}


__device__ 
void gpuBCFreeSlipSB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 10)];
    //Dead Pop are: [17, 18]
}


__device__ 
void gpuBCFreeSlipWF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 18)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 11)];
    //Dead Pop are: [9, 10]
}


__device__ 
void gpuBCFreeSlipWB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 12)];
    //Dead Pop are: [15, 16]
}


__device__ 
void gpuBCFreeSlipEF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 11)];
    //Dead Pop are: [15, 16]
}


__device__ 
void gpuBCFreeSlipEB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 17)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 12)];
    //Dead Pop are: [9, 10]
}