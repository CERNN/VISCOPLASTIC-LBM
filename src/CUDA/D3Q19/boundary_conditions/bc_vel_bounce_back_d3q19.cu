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

#include "bc_vel_bounce_back_d3q19.cuh"


__device__
void gpu_bc_vel_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];

	f[index_pop_d3q19(x, y, z, 4)] = f[index_pop_d3q19(x, y, z, 3)] - 6 * rho_w*W_1*(uy_w);
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)] - 6 * rho_w*W_2*(uy_w + ux_w);
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)] - 6 * rho_w*W_2*(uy_w + uz_w);
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)] - 6 * rho_w*W_2*(uy_w - ux_w);
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)] - 6 * rho_w*W_2*(uy_w - uz_w);


}


__device__
void gpu_bc_vel_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];

	f[index_pop_d3q19(x, y, z, 3)] = f[index_pop_d3q19(x, y, z, 4)] - 6 * rho_w*W_1*(-uy_w);
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)] - 6 * rho_w*W_2*(-uy_w - ux_w);
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)] - 6 * rho_w*W_2*(-uy_w - uz_w);
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)] - 6 * rho_w*W_2*(-uy_w + ux_w);
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)] - 6 * rho_w*W_2*(-uy_w + uz_w);
}


__device__
void gpu_bc_vel_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];
	
	f[index_pop_d3q19(x, y, z, 1)] = f[index_pop_d3q19(x, y, z, 2)] - 6 * rho_w*W_1*(-ux_w);
	f[index_pop_d3q19(x, y, z, 7)] = f[index_pop_d3q19(x, y, z, 8)] - 6 * rho_w*W_2*(-ux_w - uy_w);
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)] - 6 * rho_w*W_2*(-ux_w - uz_w);
	f[index_pop_d3q19(x, y, z, 13)] = f[index_pop_d3q19(x, y, z, 14)] - 6 * rho_w*W_2*(-ux_w + uy_w);
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)] - 6 * rho_w*W_2*(-ux_w + uz_w);
}


__device__
void gpu_bc_vel_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];

	f[index_pop_d3q19(x, y, z, 2)] = f[index_pop_d3q19(x, y, z, 1)] - 6 * rho_w*W_1*(ux_w);
	f[index_pop_d3q19(x, y, z, 8)] = f[index_pop_d3q19(x, y, z, 7)] - 6 * rho_w*W_2*(ux_w + uy_w);
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)] - 6 * rho_w*W_2*(ux_w + uz_w);
	f[index_pop_d3q19(x, y, z, 14)] = f[index_pop_d3q19(x, y, z, 13)] - 6 * rho_w*W_2*(ux_w - uy_w);
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)] - 6 * rho_w*W_2*(ux_w - uz_w);
}


__device__
void gpu_bc_vel_bounce_back_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];

	f[index_pop_d3q19(x, y, z, 6)] = f[index_pop_d3q19(x, y, z, 5)] - 6 * rho_w*W_1*(uz_w);
	f[index_pop_d3q19(x, y, z, 10)] = f[index_pop_d3q19(x, y, z, 9)] - 6 * rho_w*W_2*(uz_w + ux_w);
	f[index_pop_d3q19(x, y, z, 12)] = f[index_pop_d3q19(x, y, z, 11)] - 6 * rho_w*W_2*(uz_w + uy_w);
	f[index_pop_d3q19(x, y, z, 15)] = f[index_pop_d3q19(x, y, z, 16)] - 6 * rho_w*W_2*(uz_w - ux_w);
	f[index_pop_d3q19(x, y, z, 17)] = f[index_pop_d3q19(x, y, z, 18)] - 6 * rho_w*W_2*(uz_w - uy_w);

}


__device__
void gpu_bc_vel_bounce_back_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
	// uses node's rho as the wall's rho
	const dfloat rho_w = f[index_pop_d3q19(x, y, z, 0)] + f[index_pop_d3q19(x, y, z, 1)] + f[index_pop_d3q19(x, y, z, 2)] +
		f[index_pop_d3q19(x, y, z, 3)] + f[index_pop_d3q19(x, y, z, 4)] + f[index_pop_d3q19(x, y, z, 5)] + f[index_pop_d3q19(x, y, z, 6)] +
		f[index_pop_d3q19(x, y, z, 7)] + f[index_pop_d3q19(x, y, z, 8)] + f[index_pop_d3q19(x, y, z, 9)] + f[index_pop_d3q19(x, y, z, 10)] +
		f[index_pop_d3q19(x, y, z, 11)] + f[index_pop_d3q19(x, y, z, 12)] + f[index_pop_d3q19(x, y, z, 13)] + f[index_pop_d3q19(x, y, z, 14)] +
		f[index_pop_d3q19(x, y, z, 15)] + f[index_pop_d3q19(x, y, z, 16)] + f[index_pop_d3q19(x, y, z, 17)] + f[index_pop_d3q19(x, y, z, 18)];

	f[index_pop_d3q19(x, y, z, 5)] = f[index_pop_d3q19(x, y, z, 6)] - 6 * rho_w*W_1*(-uz_w);
	f[index_pop_d3q19(x, y, z, 9)] = f[index_pop_d3q19(x, y, z, 10)] - 6 * rho_w*W_2*(-uz_w - ux_w);
	f[index_pop_d3q19(x, y, z, 11)] = f[index_pop_d3q19(x, y, z, 12)] - 6 * rho_w*W_2*(-uz_w - uy_w);
	f[index_pop_d3q19(x, y, z, 16)] = f[index_pop_d3q19(x, y, z, 15)] - 6 * rho_w*W_2*(-uz_w + ux_w);
	f[index_pop_d3q19(x, y, z, 18)] = f[index_pop_d3q19(x, y, z, 17)] - 6 * rho_w*W_2*(-uz_w + uy_w);
}
