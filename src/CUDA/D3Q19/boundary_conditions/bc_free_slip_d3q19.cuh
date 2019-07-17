#ifndef BC_FREE_SLIP_D3Q19_CUH
#define BC_FREE_SLIP_D3Q19_CUH

#include "./../../common/func_idx.cuh"
#include <cuda_runtime.h>

/*
*	Applies free slip boundary condition on north wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_N(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on south wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_S(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on west wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_W(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on east wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_E(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_F(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations 
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_B(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on northwest wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_NW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on northeast wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_NE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on north-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_NF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on north-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_NB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on southwest wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_SW(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on southeast wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_SE(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on south-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_SF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on south-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_SB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on west-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_WF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on west-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_WB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on east-front wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_EF(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);


/*
*	Applies free slip boundary condition on east-back wall node
*	\param f[(N_X, N_Y, N_Z, Q)]: grid of populations
*	\param x: node's x value
*	\param y: node's y value
*	\param z: node's z value
*/
__device__ 
void gpu_bc_free_slip_EB(dfloat* f, const short unsigned int x, const short unsigned int y,
	const short unsigned int z);

#endif // !BC_FREE_SLIP_D3Q19_CUH