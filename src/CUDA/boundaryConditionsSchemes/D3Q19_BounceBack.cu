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

#include "D3Q19_BounceBack.h"

__device__ 
void gpuBCBounceBackN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    fPostStream[idxPop(x, y, z, 4)] = fPostCol[idxPop(x, y, z, 3)];
    fPostStream[idxPop(x, y, z, 8)] = fPostCol[idxPop(x, y, z, 7)];
    fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)];
    fPostStream[idxPop(x, y, z, 13)] = fPostCol[idxPop(x, y, z, 14)];
    fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)];
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
    //Dead Pop are: [7, 8]
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
    //Dead Pop are: [13, 14]
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
    //Dead Pop are: [17, 18]
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
    //Dead Pop are: [11, 12]
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
    //Dead Pop are: [13, 14]
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
    //Dead Pop are: [7, 8]
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
    //Dead Pop are: [11, 12]
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
    //Dead Pop are: [17, 18]
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
    //Dead Pop are: [9, 10]
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
    //Dead Pop are: [15, 16]

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
    //Dead Pop are: [15, 16]
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
    //Dead Pop are: [9, 10]
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
    //Dead Pop are: [7, 8, 9, 10, 17, 18]
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
    //Dead Pop are: [7, 8, 11, 12, 15, 16]
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
    //Dead Pop are: [13, 14, 15, 16, 17, 18]
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
    //Dead Pop are: [9, 10, 11, 12, 13, 14]

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
    //Dead Pop are: [9, 10, 11, 12, 13, 14]
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
    //Dead Pop are: [13, 14, 15, 16, 17, 18]
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
    //Dead Pop are: [7, 8, 11, 12, 15, 16]
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
    //Dead Pop are: [7, 8, 9, 10, 17, 18]
}