#ifndef BC_VEL_NEBB_D3Q19_CUH
#define BC_VEL_NEBB_D3Q19_CUH

#include "./../../common/func_idx.cuh"
#include <cuda_runtime.h>


/*
*	Applies velocity Zou-He boundary condition on north wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*	Applies velocity Zou-He boundary condition on south wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*	Applies velocity Zou-He boundary condition on west wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*	Applies velocity Zou-He boundary condition on east wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*	Applies velocity Zou-He boundary condition on front wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*	Applies velocity Zou-He boundary condition on back wall node, given velocities
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*	\param ux_w: node's x velocity
*	\param uy_w: node's y velocity
*	\param uz_w: node's z velocity
*/
__device__
void gpu_bc_vel_nebb_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


#endif // !BC_VEL_NEBB_D3Q19_CUH