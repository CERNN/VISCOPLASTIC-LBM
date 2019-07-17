#ifndef BC_PRES_NEBB_D3Q19_CUH
#define BC_PRES_NEBB_D3Q19_CUH

#include "./../../common/func_idx.cuh"
#include <cuda_runtime.h>


/*
*	Applies pressure Zou-He boundary condition on north wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure Zou-He boundary condition on south wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure Zou-He boundary condition on west wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure Zou-He boundary condition on east wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure Zou-He boundary condition on front wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


/*
*	Applies pressure Zou-He boundary condition on back wall node, given pressure
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param rho_w: node's densisty
*/
__device__
void gpu_bc_pres_nebb_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat rho_w);


#endif // !BC_PRES_NEBB_D3Q19_CUH