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

#include "bc_pres_anti_bb_d3q19.cuh"

__device__
void gpu_bc_pres_anti_bb_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w)
{
	dfloat rho_b0, rho_b1;
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, &rho_b0, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x, y - 1, z, &rho_b1, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 4)] = -f[index_pop_d3q19(x, y, z, 3)] + 2 * aux_function_anti_bb(rho_w1, u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 8)] = -f[index_pop_d3q19(x, y, z, 7)] + 2 * aux_function_anti_bb(rho_w2, u_w[1] - u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 12)] = -f[index_pop_d3q19(x, y, z, 11)] + 2 * aux_function_anti_bb(rho_w2, u_w[1] + u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 13)] = -f[index_pop_d3q19(x, y, z, 14)] + 2 * aux_function_anti_bb(rho_w2, -u_w[0] + u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 18)] = -f[index_pop_d3q19(x, y, z, 17)] + 2 * aux_function_anti_bb(rho_w2, u_w[1] - u_w[2], p1_muu15);
}


__device__
void gpu_bc_pres_anti_bb_S(dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z, const dfloat rho_w)
{
	dfloat rho_b0, rho_b1;
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, &rho_b0, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x, y + 1, z, &rho_b1, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 3)] = -f[index_pop_d3q19(x, y, z, 4)] + 2 * aux_function_anti_bb(rho_w1, -u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 7)] = -f[index_pop_d3q19(x, y, z, 8)] + 2 * aux_function_anti_bb(rho_w2, -u_w[1] - u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 11)] = -f[index_pop_d3q19(x, y, z, 12)] + 2 * aux_function_anti_bb(rho_w2, -u_w[1] - u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 14)] = -f[index_pop_d3q19(x, y, z, 13)] + 2 * aux_function_anti_bb(rho_w2, -u_w[1] + u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 17)] = -f[index_pop_d3q19(x, y, z, 18)] + 2 * aux_function_anti_bb(rho_w2, -u_w[1] + u_w[2], p1_muu15);
}


__device__
void gpu_bc_pres_anti_bb_W(dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z, const dfloat rho_w)
{
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, NULL, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x + 1, y, z, NULL, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 1)] = -f[index_pop_d3q19(x, y, z, 2)] + 2 * aux_function_anti_bb(rho_w1, -u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 7)] = -f[index_pop_d3q19(x, y, z, 8)] + 2 * aux_function_anti_bb(rho_w2, -u_w[0] - u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 9)] = -f[index_pop_d3q19(x, y, z, 10)] + 2 * aux_function_anti_bb(rho_w2, -u_w[0] - u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 13)] = -f[index_pop_d3q19(x, y, z, 14)] + 2 * aux_function_anti_bb(rho_w2, -u_w[0] + u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 15)] = -f[index_pop_d3q19(x, y, z, 16)] + 2 * aux_function_anti_bb(rho_w2, -u_w[0] + u_w[2], p1_muu15);

}


__device__
void gpu_bc_pres_anti_bb_E(dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z, const dfloat rho_w)
{
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, NULL, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x - 1, y, z, NULL, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 2)] = -f[index_pop_d3q19(x, y, z, 1)] + 2 * aux_function_anti_bb(rho_w1, u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 8)] = -f[index_pop_d3q19(x, y, z, 7)] + 2 * aux_function_anti_bb(rho_w2, u_w[0] + u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 10)] = -f[index_pop_d3q19(x, y, z, 9)] + 2 * aux_function_anti_bb(rho_w2, u_w[0] + u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 14)] = -f[index_pop_d3q19(x, y, z, 13)] + 2 * aux_function_anti_bb(rho_w2, u_w[0] - u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 16)] = -f[index_pop_d3q19(x, y, z, 15)] + 2 * aux_function_anti_bb(rho_w2, u_w[0] - u_w[2], p1_muu15);

}


__device__
void gpu_bc_pres_anti_bb_F(dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z, const dfloat rho_w)
{
	dfloat rho_b0, rho_b1;
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, &rho_b0, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x, y, z - 1, &rho_b1, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 6)] = -f[index_pop_d3q19(x, y, z, 5)] + 2 * aux_function_anti_bb(rho_w1, u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 10)] = -f[index_pop_d3q19(x, y, z, 9)] + 2 * aux_function_anti_bb(rho_w2, u_w[2] + u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 12)] = -f[index_pop_d3q19(x, y, z, 11)] + 2 * aux_function_anti_bb(rho_w2, u_w[2] + u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 15)] = -f[index_pop_d3q19(x, y, z, 16)] + 2 * aux_function_anti_bb(rho_w2, u_w[2] - u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 17)] = -f[index_pop_d3q19(x, y, z, 18)] + 2 * aux_function_anti_bb(rho_w2, u_w[2] - u_w[1], p1_muu15);

}


__device__
void gpu_bc_pres_anti_bb_B(dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z, const dfloat rho_w)
{
	dfloat rho_b0, rho_b1;
	dfloat u_b0[3]; //ub_0(ux_b0, uy_b0, uz_b0)
	dfloat u_b1[3]; //ub_1(ux_b1, uy_b1, uz_b1)
	dfloat u_w[3];

	evaluate_macr_d3q19(f, x, y, z, &rho_b0, &u_b0[0], &u_b0[1], &u_b0[2]);
	evaluate_macr_d3q19(f, x, y, z + 1, &rho_b1, &u_b1[0], &u_b1[1], &u_b1[2]);

	// extrapolates wall's velocity given the velocity of its neighbour nodes
	for (int i = 0; i < 3; i++)
	{
		// u_w[i] = u_b0[i] + 0.5 * (u_b0[i] - u_b1[i]);
		u_w[i] = 1.5 * u_b0[i] - 0.5 * u_b1[i];
	}
	// values for function equilibrium
	const dfloat p1_muu15 = 1 - 1.5 * (u_w[0] * u_w[0] + u_w[1] * u_w[1] + u_w[2] * u_w[2]);
	const dfloat rho_w1 = rho_w * W_1;
	const dfloat rho_w2 = rho_w * W_2;

	f[index_pop_d3q19(x, y, z, 5)] = -f[index_pop_d3q19(x, y, z, 6)] + 2 * aux_function_anti_bb(rho_w1, -u_w[2], p1_muu15);
	f[index_pop_d3q19(x, y, z, 9)] = -f[index_pop_d3q19(x, y, z, 10)] + 2 * aux_function_anti_bb(rho_w2, -u_w[2] - u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 11)] = -f[index_pop_d3q19(x, y, z, 12)] + 2 * aux_function_anti_bb(rho_w2, -u_w[2] - u_w[1], p1_muu15);
	f[index_pop_d3q19(x, y, z, 16)] = -f[index_pop_d3q19(x, y, z, 15)] + 2 * aux_function_anti_bb(rho_w2, -u_w[2] + u_w[0], p1_muu15);
	f[index_pop_d3q19(x, y, z, 18)] = -f[index_pop_d3q19(x, y, z, 17)] + 2 * aux_function_anti_bb(rho_w2, -u_w[2] + u_w[1], p1_muu15);

}