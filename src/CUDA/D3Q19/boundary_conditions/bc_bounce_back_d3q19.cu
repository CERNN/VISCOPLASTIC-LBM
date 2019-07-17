/*
*	LBM-CERNN
*	Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*	This program is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation; either version 2 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License along
*	with this program; if not, write to the Free Software Foundation, Inc.,
*	51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*	Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#include "bc_bounce_back_d3q19.cuh"

__device__ 
void gpu_bc_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
}


__device__ 
void gpu_bc_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
}


__device__ 
void gpu_bc_bounce_back_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_NW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_NE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_NF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
}


__device__ 
void gpu_bc_bounce_back_NB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_SW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}



__device__ 
void gpu_bc_bounce_back_SE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_SF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_SB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
}


__device__ 
void gpu_bc_bounce_back_WF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_WB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)];
}


__device__ 
void gpu_bc_bounce_back_EF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_EB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)];
}


__device__ 
void gpu_bc_bounce_back_NWF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_NW(f, x, y, z);
	gpu_bc_bounce_back_F(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_NWB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_NW(f, x, y, z);
	gpu_bc_bounce_back_B(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_NEF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_NE(f, x, y, z);
	gpu_bc_bounce_back_F(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_NEB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_NE(f, x, y, z);
	gpu_bc_bounce_back_B(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_SWF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_SW(f, x, y, z);
	gpu_bc_bounce_back_F(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_SWB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_SW(f, x, y, z);
	gpu_bc_bounce_back_B(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_SEF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_SE(f, x, y, z);
	gpu_bc_bounce_back_F(f, x, y, z);
}


__device__ 
void gpu_bc_bounce_back_SEB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	gpu_bc_bounce_back_SE(f, x, y, z);
	gpu_bc_bounce_back_B(f, x, y, z);
}