#ifndef BC_PRES_ANTI_BB_D3Q19_CUH
#define BC_PRES_ANTI_BB_D3Q19_CUH

#include "./../../common/func_idx.cuh"
#include <cuda_runtime.h>

/*
*	Auxiliary function to evaluate the term w_i*rho*(1 + 4.5(c_i*u_w)^2 - 1.5(u_w*u_w))
*	\param rho_w: rho*w_i
*	\param uc: u_w * c_i
*	\param p1_muu: 1 - 1.5*u_w*u_w
*	return: value of w_i*rho*(1 + 4.5(c_i*u_w) - 1.5(u_w*u_w))
*/
__device__
dfloat __forceinline__ aux_function_anti_bb(dfloat rho_w, dfloat uc, dfloat p1_muu)
{
	return rho_w * (p1_muu + 4.5*uc*uc);
}

/*
*	Applies pressure anti bounce-back boundary condition on north wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure anti bounce-back boundary condition on south wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure anti bounce-back boundary condition on west wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure anti bounce-back boundary condition on east wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure anti bounce-back boundary condition on front wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure anti bounce-back boundary condition on back wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_anti_bb_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);

#endif // !BC_PRES_ANTI_BB_D3Q19_CUH