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

#include "bc_free_slip_d3q19.cuh"


__device__ 
void gpu_bc_free_slip_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 17)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 11)];
}


__device__
void gpu_bc_free_slip_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 18)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 12)];
}


__device__ 
void gpu_bc_free_slip_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)];
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 14)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 8)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 10)];
}


__device__ 
void gpu_bc_free_slip_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)];
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 13)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 7)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 9)];
}


__device__ 
void gpu_bc_free_slip_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)];
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 16)];
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 18)];
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 9)];
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 11)];
}


__device__ 
void gpu_bc_free_slip_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)];
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 15)];
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 17)];
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 10)];
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 12)];
}


__device__ 
void gpu_bc_free_slip_NW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 14)];
	gpu_bc_free_slip_N(f, x, y, z);
	gpu_bc_free_slip_W(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 13)] = tmp;
}


__device__ 
void gpu_bc_free_slip_NE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 7)];
	gpu_bc_free_slip_N(f, x, y, z);
	gpu_bc_free_slip_E(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 8)] = tmp;
}


__device__ 
void gpu_bc_free_slip_NF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 11)];
	gpu_bc_free_slip_N(f, x, y, z);
	gpu_bc_free_slip_F(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 12)] = tmp;
}


__device__ 
void gpu_bc_free_slip_NB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 17)];
	gpu_bc_free_slip_S(f, x, y, z);
	gpu_bc_free_slip_B(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 18)] = tmp;
}


__device__ 
void gpu_bc_free_slip_SW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 8)];
	gpu_bc_free_slip_S(f, x, y, z);
	gpu_bc_free_slip_W(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 7)] = tmp;
}


__device__ 
void gpu_bc_free_slip_SE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 13)];
	gpu_bc_free_slip_S(f, x, y, z);
	gpu_bc_free_slip_E(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 14)] = tmp;
}


__device__ 
void gpu_bc_free_slip_SF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 18)];
	gpu_bc_free_slip_S(f, x, y, z);
	gpu_bc_free_slip_F(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 17)] = tmp;
}


__device__ 
void gpu_bc_free_slip_SB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 12)];
	gpu_bc_free_slip_S(f, x, y, z);
	gpu_bc_free_slip_B(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 11)] = tmp;
}


__device__ 
void gpu_bc_free_slip_WF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 16)];
	gpu_bc_free_slip_W(f, x, y, z);
	gpu_bc_free_slip_F(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 15)] = tmp;
}


__device__ 
void gpu_bc_free_slip_WB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 10)];
	gpu_bc_free_slip_W(f, x, y, z);
	gpu_bc_free_slip_B(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 9)] = tmp;
}


__device__ 
void gpu_bc_free_slip_EF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 10)];
	gpu_bc_free_slip_E(f, x, y, z);
	gpu_bc_free_slip_F(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 9)] = tmp;
}


__device__ 
void gpu_bc_free_slip_EB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z)
{
	const dfloat tmp = f[index_pop_d3q19(x, y, z, 15)];
	gpu_bc_free_slip_E(f, x, y, z);
	gpu_bc_free_slip_B(f, x, y, z);
	f[index_pop_d3q19(x, y, z, 16)] = tmp;
}