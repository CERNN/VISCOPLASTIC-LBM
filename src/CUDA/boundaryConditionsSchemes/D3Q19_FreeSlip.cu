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
void gpuBCFreeSlipN(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(4)] = f[idxPop(x, yp1, z, 3)];
    fNode[(8)] = f[idxPop(x, yp1, z, 14)];
    fNode[(12)] = f[idxPop(x, yp1, z, 17)];
    fNode[(13)] = f[idxPop(x, yp1, z, 7)];
    fNode[(18)] = f[idxPop(x, yp1, z, 11)];
}


__device__
void gpuBCFreeSlipS(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(3)] = f[idxPop(x, ym1, z, 4)];
    fNode[(7)] = f[idxPop(x, ym1, z, 13)];
    fNode[(11)] = f[idxPop(x, ym1, z, 18)];
    fNode[(14)] = f[idxPop(x, ym1, z, 8)];
    fNode[(17)] = f[idxPop(x, ym1, z, 12)];
}


__device__ 
void gpuBCFreeSlipW(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(1)] = f[idxPop(xm1, y, z, 2)];
    fNode[(7)] = f[idxPop(xm1, y, z, 14)];
    fNode[(9)] = f[idxPop(xm1, y, z, 16)];
    fNode[(13)] = f[idxPop(xm1, y, z, 8)];
    fNode[(15)] = f[idxPop(xm1, y, z, 10)];
}


__device__ 
void gpuBCFreeSlipE(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(2)] = f[idxPop(xp1, y, z, 1)];
    fNode[(8)] = f[idxPop(xp1, y, z, 13)];
    fNode[(10)] = f[idxPop(xp1, y, z, 15)];
    fNode[(14)] = f[idxPop(xp1, y, z, 7)];
    fNode[(16)] = f[idxPop(xp1, y, z, 9)];
}


__device__ 
void gpuBCFreeSlipF(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(6)] = f[idxPop(x, y, zp1, 5)];
    fNode[(10)] = f[idxPop(x, y, zp1, 16)];
    fNode[(12)] = f[idxPop(x, y, zp1, 18)];
    fNode[(15)] = f[idxPop(x, y, zp1, 9)];
    fNode[(17)] = f[idxPop(x, y, zp1, 11)];
}


__device__ 
void gpuBCFreeSlipB(dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fNode[(5)] = f[idxPop(x, y, zm1, 6)];
    fNode[(9)] = f[idxPop(x, y, zm1, 15)];
    fNode[(11)] = f[idxPop(x, y, zm1, 17)];
    fNode[(16)] = f[idxPop(x, y, zm1, 10)];
    fNode[(18)] = f[idxPop(x, y, zm1, 12)];
}