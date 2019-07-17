#ifndef BC_BOUNCE_BACK_D3Q19_CUH
#define BC_BOUNCE_BACK_D3Q19_CUH

#include "./../../common/func_idx.cuh"
#include <cuda_runtime.h>


/*
*	Applies bounce back boundary condition on north wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on south wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on west wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on east wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northwest wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northeast wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on north-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on north-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southwest wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southeast wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on south-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on south-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on west-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_WF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on west-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_WB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on east-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_EF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on east-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_EB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northwest-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NWF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northwest-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NWB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northeast-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NEF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on northeast-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_NEB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southwest-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SWF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southwest-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SWB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southeast-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SEF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies bounce back boundary condition on southeast-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_bounce_back_SEB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


#endif // !BC_BOUNCE_BACK_D3Q19_CUH
