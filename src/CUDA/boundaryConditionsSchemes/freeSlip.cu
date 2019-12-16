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

#include "freeSlip.h"


__device__ 
void gpuBCFreeSlipN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(xp1, y, z, 14)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, zp1, 17)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(xm1, y, z, 7)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, zm1, 11)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(xp1, y, zp1, 24)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(xp1, y, zm1, 25)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(xm1, y, zm1, 19)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(xm1, y, zp1, 21)];
    #endif
}


__device__
void gpuBCFreeSlipS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    //const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    //const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(xm1, y, z, 13)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, zm1, 18)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(xp1, y, z, 8)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, zp1, 12)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(xm1, y, zm1, 23)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(xm1, y, zp1, 26)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(xp1, y, zp1, 20)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(xp1, y, zm1, 22)];
    #endif
}


__device__ 
void gpuBCFreeSlipW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, ym1, z, 14)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, zm1, 16)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, yp1, z, 8)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, zp1, 10)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, ym1, zm1, 25)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, ym1, zp1, 24)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, yp1, zm1, 22)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, yp1, zp1, 20)];
    #endif
}


__device__ 
void gpuBCFreeSlipE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    //const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    //const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, yp1, z, 13)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, zp1, 15)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, ym1, z, 7)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, zm1, 9)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, yp1, zp1, 26)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, yp1, zm1, 23)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, ym1, zp1, 21)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, ym1, zm1, 19)];
    #endif
}


__device__ 
void gpuBCFreeSlipF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(xp1, y, z, 16)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, yp1, z, 18)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(xm1, y, z, 9)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, ym1, z, 11)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(xp1, yp1, z, 22)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(xm1, ym1, z, 19)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(xp1, ym1, z, 25)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(xm1, yp1, z, 23)];
    #endif
}


__device__ 
void gpuBCFreeSlipB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    //const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    //const unsigned short int zm1 = (NZ + z - 1) % NZ;
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(xm1, y, z, 15)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, ym1, z, 17)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(xp1, y, z, 10)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, yp1, z, 12)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(xm1, ym1, z, 21)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(xp1, yp1, z, 20)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(xm1, yp1, z, 26)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(xp1, ym1, z, 24)];
    #endif
}