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

#include "D3Q27_FreeSlip.h"


__device__ 
void gpuBCFreeSlipN(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(xp1, y, z, 14)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, zp1, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(xm1, y, z, 7)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, zm1, 11)];
    f[idxPop(x, y, z, 20)] = f[idxPop(xp1, y, zp1, 24)];
    f[idxPop(x, y, z, 22)] = f[idxPop(xp1, y, zm1, 25)];
    f[idxPop(x, y, z, 23)] = f[idxPop(xm1, y, zm1, 19)];
    f[idxPop(x, y, z, 26)] = f[idxPop(xm1, y, zp1, 21)];
}


__device__
void gpuBCFreeSlipS(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(xm1, y, z, 13)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, zm1, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(xp1, y, z, 8)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, zp1, 12)];
    f[idxPop(x, y, z, 19)] = f[idxPop(xm1, y, zm1, 23)];
    f[idxPop(x, y, z, 21)] = f[idxPop(xm1, y, zp1, 26)];
    f[idxPop(x, y, z, 24)] = f[idxPop(xp1, y, zp1, 20)];
    f[idxPop(x, y, z, 25)] = f[idxPop(xp1, y, zm1, 22)];
}


__device__ 
void gpuBCFreeSlipW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, ym1, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, zm1, 16)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, yp1, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, zp1, 10)];
    f[idxPop(x, y, z, 19)] = f[idxPop(x, ym1, zm1, 25)];
    f[idxPop(x, y, z, 21)] = f[idxPop(x, ym1, zp1, 24)];
    f[idxPop(x, y, z, 23)] = f[idxPop(x, yp1, zm1, 22)];
    f[idxPop(x, y, z, 26)] = f[idxPop(x, yp1, zp1, 20)];
}


__device__ 
void gpuBCFreeSlipE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, yp1, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, zp1, 15)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, ym1, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, zm1, 9)];
    f[idxPop(x, y, z, 20)] = f[idxPop(x, yp1, zp1, 26)];
    f[idxPop(x, y, z, 22)] = f[idxPop(x, yp1, zm1, 23)];
    f[idxPop(x, y, z, 24)] = f[idxPop(x, ym1, zp1, 21)];
    f[idxPop(x, y, z, 25)] = f[idxPop(x, ym1, zm1, 19)];
}


__device__ 
void gpuBCFreeSlipF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 10)] = f[idxPop(xp1, y, z, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, yp1, z, 18)];
    f[idxPop(x, y, z, 15)] = f[idxPop(xm1, y, z, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, ym1, z, 11)];
    f[idxPop(x, y, z, 20)] = f[idxPop(xp1, yp1, z, 22)];
    f[idxPop(x, y, z, 21)] = f[idxPop(xm1, ym1, z, 19)];
    f[idxPop(x, y, z, 24)] = f[idxPop(xp1, ym1, z, 25)];
    f[idxPop(x, y, z, 26)] = f[idxPop(xm1, yp1, z, 23)];
}


__device__ 
void gpuBCFreeSlipB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 9)] = f[idxPop(xm1, y, z, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, ym1, z, 17)];
    f[idxPop(x, y, z, 16)] = f[idxPop(xp1, y, z, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, yp1, z, 12)];
    f[idxPop(x, y, z, 19)] = f[idxPop(xm1, ym1, z, 21)];
    f[idxPop(x, y, z, 22)] = f[idxPop(xp1, yp1, z, 20)];
    f[idxPop(x, y, z, 23)] = f[idxPop(xm1, yp1, z, 26)];
    f[idxPop(x, y, z, 25)] = f[idxPop(xp1, ym1, z, 24)];
}


__device__ 
void gpuBCFreeSlipNW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, zm1, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, zp1, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, zp1, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, zm1, 11)];
    f[idxPop(x, y, z, 23)] = f[idxPop(x, y, zm1, 25)];
    f[idxPop(x, y, z, 26)] = f[idxPop(x, y, zp1, 24)];
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCFreeSlipNE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, zp1, 15)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, zp1, 17)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, zm1, 9)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, zm1, 11)];
    f[idxPop(x, y, z, 20)] = f[idxPop(x, y, zp1, 21)];
    f[idxPop(x, y, z, 22)] = f[idxPop(x, y, zm1, 19)];
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}


__device__ 
void gpuBCFreeSlipNF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(xp1, y, z, 14)];
    f[idxPop(x, y, z, 10)] = f[idxPop(xp1, y, z, 16)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(xm1, y, z, 7)];
    f[idxPop(x, y, z, 15)] = f[idxPop(xm1, y, z, 9)];
    f[idxPop(x, y, z, 20)] = f[idxPop(xp1, y, z, 25)];
    f[idxPop(x, y, z, 26)] = f[idxPop(xm1, y, z, 19)];
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCFreeSlipNB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(xp1, y, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(xm1, y, z, 15)];
    f[idxPop(x, y, z, 13)] = f[idxPop(xm1, y, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(xp1, y, z, 10)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    f[idxPop(x, y, z, 22)] = f[idxPop(xp1, y, z, 24)];
    f[idxPop(x, y, z, 23)] = f[idxPop(xm1, y, z, 21)];
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCFreeSlipSW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, zm1, 16)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, zm1, 18)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, zp1, 10)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, zp1, 12)];
    f[idxPop(x, y, z, 19)] = f[idxPop(x, y, zm1, 22)];
    f[idxPop(x, y, z, 21)] = f[idxPop(x, y, zp1, 20)];
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}


__device__ 
void gpuBCFreeSlipSE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, zp1, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, zm1, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, zm1, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, zp1, 12)];
    f[idxPop(x, y, z, 24)] = f[idxPop(x, y, zp1, 26)];
    f[idxPop(x, y, z, 25)] = f[idxPop(x, y, zm1, 23)];
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCFreeSlipSF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(xm1, y, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(xp1, y, z, 16)];
    f[idxPop(x, y, z, 14)] = f[idxPop(xp1, y, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(xm1, y, z, 9)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    f[idxPop(x, y, z, 21)] = f[idxPop(xm1, y, z, 23)];
    f[idxPop(x, y, z, 24)] = f[idxPop(xp1, y, z, 22)];
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCFreeSlipSB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(xm1, y, z, 13)];
    f[idxPop(x, y, z, 9)] = f[idxPop(xm1, y, z, 15)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(xp1, y, z, 8)];
    f[idxPop(x, y, z, 16)] = f[idxPop(xp1, y, z, 10)];
    f[idxPop(x, y, z, 19)] = f[idxPop(xm1, y, z, 26)];
    f[idxPop(x, y, z, 25)] = f[idxPop(xp1, y, z, 20)];
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCFreeSlipWF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, ym1, z, 14)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, yp1, z, 18)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, yp1, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, ym1, z, 11)];
    f[idxPop(x, y, z, 21)] = f[idxPop(x, ym1, z, 25)];
    f[idxPop(x, y, z, 26)] = f[idxPop(x, yp1, z, 22)];
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}


__device__ 
void gpuBCFreeSlipWB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, ym1, z, 14)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, ym1, z, 17)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, yp1, z, 8)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, yp1, z, 12)];
    f[idxPop(x, y, z, 19)] = f[idxPop(x, ym1, z, 24)];
    f[idxPop(x, y, z, 23)] = f[idxPop(x, yp1, z, 20)];
    //Dead Pop are: [15, 16, 21, 22, 25, 26]
}


__device__ 
void gpuBCFreeSlipEF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, yp1, z, 13)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, yp1, z, 18)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, ym1, z, 7)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, ym1, z, 11)];
    f[idxPop(x, y, z, 20)] = f[idxPop(x, yp1, z, 23)];
    f[idxPop(x, y, z, 24)] = f[idxPop(x, ym1, z, 19)];
    //Dead Pop are: [15, 16, 21, 22, 25, 26]
}


__device__ 
void gpuBCFreeSlipEB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, yp1, z, 13)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, ym1, z, 17)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, ym1, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, yp1, z, 12)];
    f[idxPop(x, y, z, 22)] = f[idxPop(x, yp1, z, 26)];
    f[idxPop(x, y, z, 25)] = f[idxPop(x, ym1, z, 21)];
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}