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

#include "bounceBack.h"

#ifdef BC_SCHEME_BOUNCE_BACK

__device__ 
void gpuBCBounceBackN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
}


__device__ 
void gpuBCBounceBackS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
}


__device__ 
void gpuBCBounceBackW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
}


__device__ 
void gpuBCBounceBackE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
}


__device__ 
void gpuBCBounceBackF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
}


__device__ 
void gpuBCBounceBackB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
}


__device__ 
void gpuBCBounceBackNW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCBounceBackNE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    #endif
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackNF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackNB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    #endif
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCBounceBackSW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    #endif
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}



__device__ 
void gpuBCBounceBackSE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCBounceBackSF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    #endif
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCBounceBackSB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}


__device__ 
void gpuBCBounceBackWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    #endif
    //Dead Pop are: [15, 16, 21, 22, 25, 26]

}


__device__ 
void gpuBCBounceBackEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    #endif
    //Dead Pop are: [15, 16, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}


__device__ 
void gpuBCBounceBackNWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 26)] = fPostCol[idxPop(x, y, z, 25)];
    #endif
    //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackNWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 23)] = fPostCol[idxPop(x, y, z, 24)];
    #endif
    //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackNEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 20)] = fPostCol[idxPop(x, y, z, 19)];
    #endif
    //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackNEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 22)] = fPostCol[idxPop(x, y, z, 21)];
    #endif
    //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]

}


__device__ 
void gpuBCBounceBackSWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 21)] = fPostCol[idxPop(x, y, z, 22)];
    #endif
    //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackSWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 1)] = fPostCol[idxPop(x, y, z, 2)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 7)] = fPostCol[idxPop(x, y, z, 8)];
    fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 19)] = fPostCol[idxPop(x, y, z, 20)];
    #endif
    //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackSEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 6)] = fPostCol[idxPop(x, y, z, 5)];
    fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 24)] = fPostCol[idxPop(x, y, z, 23)];
    #endif
    //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackSEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 2)] = fPostCol[idxPop(x, y, z, 1)];
    fPostStream[idxPop(x, y, z, 3)] = fPostCol[idxPop(x, y, z, 4)];
    fPostStream[idxPop(x, y, z, 5)] = fPostCol[idxPop(x, y, z, 6)];
    fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)];
    fPostStream[idxPop(x, y, z, 14)] = fPostCol[idxPop(x, y, z, 13)];
    fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)];
    #ifdef D3Q27
    fPostStream[idxPop(x, y, z, 25)] = fPostCol[idxPop(x, y, z, 26)];
    #endif
    //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
}

#endif