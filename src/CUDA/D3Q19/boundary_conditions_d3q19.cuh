#ifndef BOUNDARY_CONDITIONS_D3Q19_CUH
#define BOUNDARY_CONDITIONS_D3Q19_CUH

//TODO: 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "boundary_conditions/bc_bounce_back_d3q19.cuh"
#include "boundary_conditions/bc_free_slip_d3q19.cuh"
#include "boundary_conditions/bc_vel_bounce_back_d3q19.cuh"
#include "boundary_conditions/bc_vel_nebb_d3q19.cuh"
#include "boundary_conditions/bc_pres_anti_bb_d3q19.cuh"
#include "boundary_conditions/bc_pres_nebb_d3q19.cuh"
#include "structs.cuh"

/*
*   Builds boundary conditions map
*   \param map_bc_gpu: device pointer to the boundary conditions map
*   \param BC_TYPE: type of boundary condition (options in "var.h" defines)
*/
__host__
void build_boundary_conditions(NodeTypeMap* const map_bc_gpu, const int BC_TYPE);


/*
*   Builds boundary conditions map for lid driven cavity (N: z velocity bounce-back; E, W: periodic)
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_lid_driven_cavity(NodeTypeMap* const map_bc_gpu);


/*
*   Builds boundary conditions map for parallel plates (N, S: stationary walls; W, E: pressure condition; F, B: periodic (BC_NULL))
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_parallel_plates(NodeTypeMap* const map_bc_gpu);


/*
*   Builds boundary conditions map for square duct (N, S, W, E: stationary walls; F, B: periodic condition; ADD FORCE SCHEME)
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_square_duct(NodeTypeMap* const map_bc_gpu);


/*
*   Builds boundary conditions map for taylor green vortex (all periodic)
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_taylor_green_vortex(NodeTypeMap* const map_bc_gpu);


/*
*   Builds boundary conditions map to perform tests
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_TESTES(NodeTypeMap* const map_bc_gpu);


/*
*   Applies boundary conditions given node type and its population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_boundary_conditions(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies bounce back boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_bounce_back(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies velocity bounce back boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_vel_bounce_back(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies free slip boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_free_slip(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies pressure anti bounce back boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_pres_anti_bb(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies pressure non-equilibrium bounce back boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_pres_nebb(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);


/*
*   Applies velocity Zou-He boundary condition given node's population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, N_Z, Q)]: grid of populations from 0 to 19
*   \param x: node's x value
*   \param y: node's y value
*   \param z: node's z value
*/
__device__
void gpu_sch_vel_zouhe(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z);

#endif // !BOUNDARY_CONDITIONS_D3Q19_CUH