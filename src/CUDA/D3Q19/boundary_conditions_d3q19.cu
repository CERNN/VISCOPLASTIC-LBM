/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#include "boundary_conditions_d3q19.cuh"

__host__ 
void build_boundary_conditions(NodeTypeMap* const map_bc_gpu, const int BC_TYPE)
{
    // blocks in grid
    dim3 grid(N_X/nThreads_X, N_Y/nThreads_Y, N_Z/nThreads_Z);
    // threads in block
    dim3 threads(nThreads_X, nThreads_Y, nThreads_Z);

    switch (BC_TYPE)
    {
    case BC_LID_DRIVEN_CAVITY:
        gpu_build_boundary_conditions_lid_driven_cavity<<<grid, threads>>>(map_bc_gpu);
        break;
    case BC_PARALLEL_PLATES:
        gpu_build_boundary_conditions_parallel_plates<<<grid, threads>>>(map_bc_gpu);
        break;
    case BC_SQUARE_DUCT:
        gpu_build_boundary_conditions_square_duct<<<grid, threads>>>(map_bc_gpu);
        break;
    case BC_TAYLOR_GREEN_VORTEX:
        gpu_build_boundary_conditions_taylor_green_vortex<<<grid, threads>>>(map_bc_gpu);
        break;
    default:
        gpu_build_boundary_conditions_TESTES<<<grid, threads>>>(map_bc_gpu);
        break;
    }
}

__global__
void gpu_build_boundary_conditions_lid_driven_cavity(NodeTypeMap* const map_bc_gpu)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    map_bc_gpu[index_scalar_d3(x, y, z)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_NULL);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_geometry(CONCAVE);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(0); // manually assigned (index of uz=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)

    if(y == 0 && x == 0 && z == 0) // SWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);
    }
    else if(y == 0 && x == 0 && z == (N_Z-1)) // SWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if(y == 0 && x == (N_X-1) && z == 0) // SEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);

    }
    else if(y == 0 && x == (N_X-1) && z == (N_Z-1)) // SEF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if(y == (N_Y-1) && x == 0 && z == 0) // NWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);
    }
    else if(y == (N_Y-1) && x == 0 && z == (N_Z-1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if(y == (N_Y-1) && x == (N_X-1) && z == 0) // NEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);

    }
    else if(y == (N_Y-1) && x == (N_X-1) && z == (N_Z-1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if(y == 0 && x == 0) // SW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if(y == 0 && x == (N_X-1)) // SE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if(y == (N_Y-1) && x == 0) // NW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if(y == (N_Y-1) && x == (N_X-1)) // NE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if(y == 0 && z == 0) // SB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);
    }
    else if(y == 0 && z == (N_Z-1)) // SF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if(y == (N_Y-1) && z == 0) // NB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);
    }
    else if(y == (N_Y-1) && z == (N_Z-1)) // NF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if(x == 0 && z == 0) // WB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if(x == 0 && z == (N_Z-1)) // WF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
    else if(x == (N_X-1) && z == 0) // EB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if(x == (N_X-1) && z == (N_Z-1)) // EF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
    else if(y == 0) // S
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if(y == (N_Y-1)) // N
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if(x == 0) // W
    {
        
    }
    else if(x == (N_X-1)) // E
    {

    }
    else if(z == 0) // B
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if(z == (N_Z-1)) // F
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
}


__global__
void gpu_build_boundary_conditions_parallel_plates(NodeTypeMap* const map_bc_gpu)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;

    map_bc_gpu[index_scalar_d3(x, y, z)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_NULL);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_geometry(CONCAVE);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(0); // manually assigned (index of uz=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)

    if (y == 0 && x == 0 && z == 0) // SWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && x == 0 && z == (N_Z - 1)) // SWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && x == (N_X - 1) && z == 0) // SEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && x == (N_X - 1) && z == (N_Z - 1)) // SEF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1) && x == 0 && z == 0) // NWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == (N_Y - 1) && x == 0 && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == 0) // NEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);

    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == 0 && x == 0) // SW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && x == (N_X - 1)) // SE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1) && x == 0) // NW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == (N_Y - 1) && x == (N_X - 1)) // NE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == 0 && z == 0) // SB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && z == (N_Z - 1)) // SF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1) && z == 0) // NB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == (N_Y - 1) && z == (N_Z - 1)) // NF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (x == 0 && z == 0) // WB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(1); // manually assigned (index of rho_w = RHO_IN)
    }
    else if (x == 0 && z == (N_Z - 1)) // WF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(2); // manually assigned (index of rho_w = RHO_IN)
    }
    else if (x == (N_X - 1) && z == 0) // EB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(1); // manually assigned (index of rho_w = RHO_OUT)
    }
    else if (x == (N_X - 1) && z == (N_Z - 1)) // EF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(2); // manually assigned (index of rho_w = RHO_OUT)
    }
    else if (y == 0) // S
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1)) // N
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_VEL_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (x == 0) // W
    {

    }
    else if (x == (N_X - 1)) // E
    {

    }
    else if (z == 0) // B
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(1); // manually assigned (index of rho_w = RHO_IN)
    }
    else if (z == (N_Z - 1)) // F
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(2); // manually assigned (index of rho_w = RHO_OUT)
    }
}


__global__
void gpu_build_boundary_conditions_square_duct(NodeTypeMap * const map_bc_gpu)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;

    map_bc_gpu[index_scalar_d3(x, y, z)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_NULL);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_geometry(CONCAVE);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(0); // manually assigned (index of uz=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)

    if (y == 0 && x == 0 && z == 0) // SWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_WEST);
    }
    else if (y == 0 && x == 0 && z == (N_Z - 1)) // SWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_WEST);
    }
    else if (y == 0 && x == (N_X - 1) && z == 0) // SEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_EAST);
    }
    else if (y == 0 && x == (N_X - 1) && z == (N_Z - 1)) // SEF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_EAST);
    }
    else if (y == (N_Y - 1) && x == 0 && z == 0) // NWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_WEST);
    }
    else if (y == (N_Y - 1) && x == 0 && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_WEST);
    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == 0) // NEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_EAST);

    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_EAST);
    }
    else if (y == 0 && x == 0) // SW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_WEST);
    }
    else if (y == 0 && x == (N_X - 1)) // SE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_EAST);
    }
    else if (y == (N_Y - 1) && x == 0) // NW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_WEST);
    }
    else if (y == (N_Y - 1) && x == (N_X - 1)) // NE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_EAST);
    }
    else if (y == 0 && z == 0) // SB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && z == (N_Z - 1)) // SF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1) && z == 0) // NB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (y == (N_Y - 1) && z == (N_Z - 1)) // NF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (x == 0 && z == 0) // WB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(WEST);
    }
    else if (x == 0 && z == (N_Z - 1)) // WF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(WEST);
    }
    else if (x == (N_X - 1) && z == 0) // EB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(EAST);
    }
    else if (x == (N_X - 1) && z == (N_Z - 1)) // EF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(EAST);
    }
    else if (y == 0) // S
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1)) // N
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
    }
    else if (x == 0) // W
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(WEST);
    }
    else if (x == (N_X - 1)) // E
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(EAST);
    }
    else if (z == 0) // B
    {
    
    }
    else if (z == (N_Z - 1)) // F
    {
    
    }
}


__global__
void gpu_build_boundary_conditions_taylor_green_vortex(NodeTypeMap* const map_bc_gpu)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;

    map_bc_gpu[index_scalar_d3(x, y, z)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_NULL);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_geometry(CONCAVE);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(0); // manually assigned (index of uz=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)
}


__global__
void gpu_build_boundary_conditions_TESTES(NodeTypeMap * const map_bc_gpu)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;

    map_bc_gpu[index_scalar_d3(x, y, z)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_NULL);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_geometry(CONCAVE);
    map_bc_gpu[index_scalar_d3(x, y, z)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(0); // manually assigned (index of uz=0)
    map_bc_gpu[index_scalar_d3(x, y, z)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)

    if (y == 0 && x == 0 && z == 0) // SWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);
    }
    else if (y == 0 && x == 0 && z == (N_Z - 1)) // SWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if (y == 0 && x == (N_X - 1) && z == 0) // SEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);

    }
    else if (y == 0 && x == (N_X - 1) && z == (N_Z - 1)) // SEF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if (y == (N_Y - 1) && x == 0 && z == 0) // NWB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);
    }
    else if (y == (N_Y - 1) && x == 0 && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == 0) // NEB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);

    }
    else if (y == (N_Y - 1) && x == (N_X - 1) && z == (N_Z - 1)) // NWF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if (y == 0 && x == 0) // SW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == 0 && x == (N_X - 1)) // SE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1) && x == 0) // NW
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if (y == (N_Y - 1) && x == (N_X - 1)) // NE
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if (y == 0 && z == 0) // SB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_BACK);
    }
    else if (y == 0 && z == (N_Z - 1)) // SF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH_FRONT);
    }
    else if (y == (N_Y - 1) && z == 0) // NB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_BACK);
    }
    else if (y == (N_Y - 1) && z == (N_Z - 1)) // NF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH_FRONT);
    }
    else if (x == 0 && z == 0) // WB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if (x == 0 && z == (N_Z - 1)) // WF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
    else if (x == (N_X - 1) && z == 0) // EB
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if (x == (N_X - 1) && z == (N_Z - 1)) // EF
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
    else if (y == 0) // S
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(SOUTH);
    }
    else if (y == (N_Y - 1)) // N
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(NORTH);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_uz_idx(1); // manually assigned (index of ux=U_MAX)
    }
    else if (x == 0) // W
    {

    }
    else if (x == (N_X - 1)) // E
    {

    }
    else if (z == 0) // B
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(BACK);
    }
    else if (z == (N_Z - 1)) // F
    {
        map_bc_gpu[index_scalar_d3(x, y, z)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar_d3(x, y, z)].set_direction(FRONT);
    }
}


__device__
void gpu_boundary_conditions(NodeTypeMap* nt_gpu, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    /*
    -> BC_SCHEME
        -> DIRECTION
            -> GEOMETRY
    */
    switch(nt_gpu->get_BC_scheme())
    {
    case BC_NULL:
        return;
    case BC_SCHEME_BOUNCE_BACK:
        gpu_sch_bounce_back(nt_gpu, f, x, y, z);
        break;
    case BC_SCHEME_FREE_SLIP:
        gpu_sch_free_slip(nt_gpu, f, x, y, z);
        break;
    case BC_SCHEME_VEL_BOUNCE_BACK:
        gpu_sch_vel_bounce_back(nt_gpu, f, x, y, z);
        break;  
    case BC_SCHEME_VEL_NEBB:
        gpu_sch_vel_zouhe(nt_gpu, f, x, y, z);
            break;
    case BC_SCHEME_PRES_ANTI_BB:
        gpu_sch_pres_anti_bb(nt_gpu, f, x, y, z);
        break;
    case BC_SCHEME_PRES_NEBB:
        gpu_sch_pres_nebb(nt_gpu, f, x, y, z);
    default:
        break;
    }
}


__device__
void gpu_sch_bounce_back(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch(nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_bounce_back_N(f, x, y, z);
        break;

    case SOUTH:
        gpu_bc_bounce_back_S(f, x, y, z);
        break;

    case WEST:
        gpu_bc_bounce_back_W(f, x, y, z);
        break;

    case EAST:
        gpu_bc_bounce_back_E(f, x, y, z);
        break;

    case FRONT:
        gpu_bc_bounce_back_F(f, x, y, z);
        break;

    case BACK:
        gpu_bc_bounce_back_B(f, x, y, z);
        break;

    case NORTH_WEST:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NW(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_EAST:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NE(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NF(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NB(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_WEST:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SW(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_EAST:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SE(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SF(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SB(f, x, y, z);
        else
            int a = 0;
        break;

    case WEST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_WF(f, x, y, z);
        else
            int a = 0;
        break;

    case WEST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_WB(f, x, y, z);
        else
            int a = 0;
        break;

    case EAST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_EF(f, x, y, z);
        else
            int a = 0;
        break;

    case EAST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_EB(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_WEST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NWF(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_WEST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NWB(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_EAST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NEF(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_EAST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_NEB(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_WEST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SWF(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_WEST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SWB(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_EAST_FRONT:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SEF(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_EAST_BACK:
        if(nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_bounce_back_SEB(f, x, y, z);
        else
            int a = 0; //
        break;

    default:
        break;
    }
}


__device__
void gpu_sch_vel_bounce_back(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_vel_bounce_back_N(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case SOUTH:
        gpu_bc_vel_bounce_back_S(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case WEST:
        gpu_bc_vel_bounce_back_W(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case EAST:
        gpu_bc_vel_bounce_back_E(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case FRONT:
        gpu_bc_vel_bounce_back_F(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case BACK:
        gpu_bc_vel_bounce_back_B(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;
    default:
        break;
    }
}


__device__
void gpu_sch_free_slip(NodeTypeMap* nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_free_slip_N(f, x, y, z);
        break;

    case SOUTH:
        gpu_bc_free_slip_S(f, x, y, z);
        break;

    case WEST:
        gpu_bc_free_slip_W(f, x, y, z);
        break;

    case EAST:
        gpu_bc_free_slip_E(f, x, y, z);
        break;

    case FRONT:
        gpu_bc_free_slip_F(f, x, y, z);
        break;

    case BACK:
        gpu_bc_free_slip_B(f, x, y, z);
        break;

    case NORTH_WEST:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_NW(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_EAST:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_NE(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_FRONT:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_NF(f, x, y, z);
        else
            int a = 0;
        break;

    case NORTH_BACK:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_NB(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_WEST:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_SW(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_EAST:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_SE(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_FRONT:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_SF(f, x, y, z);
        else
            int a = 0;
        break;

    case SOUTH_BACK:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_SB(f, x, y, z);
        else
            int a = 0;
        break;

    case WEST_FRONT:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_WF(f, x, y, z);
        else
            int a = 0;
        break;

    case WEST_BACK:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_WB(f, x, y, z);
        else
            int a = 0;
        break;

    case EAST_FRONT:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_EF(f, x, y, z);
        else
            int a = 0;
        break;

    case EAST_BACK:
        if (nt_gpu->get_geometry() == CONCAVE)
            gpu_bc_free_slip_EB(f, x, y, z);
        else
            int a = 0;
        break;
    default:
        break;
    }
}


__device__
void gpu_sch_pres_anti_bb(NodeTypeMap* nt_gpu, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_pres_anti_bb_N(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case SOUTH:
        gpu_bc_pres_anti_bb_S(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case WEST:
        gpu_bc_pres_anti_bb_W(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case EAST:
        gpu_bc_pres_anti_bb_E(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case FRONT:
        gpu_bc_pres_anti_bb_F(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case BACK:
        gpu_bc_pres_anti_bb_B(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;
    default:
        break;
    }
}


__device__
void gpu_sch_pres_nebb(NodeTypeMap* nt_gpu, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_pres_nebb_N(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case SOUTH:
        gpu_bc_pres_nebb_S(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case WEST:
        gpu_bc_pres_nebb_W(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case EAST:
        gpu_bc_pres_nebb_E(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case FRONT:
        gpu_bc_pres_nebb_F(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;

    case BACK:
        gpu_bc_pres_nebb_B(f, x, y, z, rho_bc[nt_gpu->get_rho_idx()]);
        break;
    default:
        break;
    }
}


__device__
void gpu_sch_vel_zouhe(NodeTypeMap * nt_gpu, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (nt_gpu->get_direction())
    {
    case NORTH:
        gpu_bc_vel_nebb_N(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()], 
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case SOUTH:
        gpu_bc_vel_nebb_S(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()],
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case WEST:
        gpu_bc_vel_nebb_W(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()],
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case EAST:
        gpu_bc_vel_nebb_E(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()],
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case FRONT:
        gpu_bc_vel_nebb_F(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()],
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;

    case BACK:
        gpu_bc_vel_nebb_B(f, x, y, z, ux_bc[nt_gpu->get_ux_idx()],
            uy_bc[nt_gpu->get_uy_idx()], uz_bc[nt_gpu->get_uz_idx()]);
        break;
    default:
        break;
    }
}