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

#include "boundary_conditions_d2q9.cuh"
#include "lbm_d2q9.cuh" // not in .cuh for no cycle include

__host__ 
void build_boundary_conditions(NodeTypeMap* const map_bc_gpu, const int BC_TYPE)
{
    // blocks in grid
    dim3 grid(N_X / nThreads_X, N_X / nThreads_Y, 1);
    // threads in block
    dim3 threads(nThreads_X, nThreads_Y, 1);

    switch (BC_TYPE)
    {
    case BC_LID_DRIVEN_CAVITY:
        gpu_build_boundary_conditions_lid_driven_cavity<<<grid, threads>>>(map_bc_gpu);
        break;
    case BC_PARALLEL_PLATES:
        gpu_build_boundary_conditions_parallel_plates<<<grid, threads>>>(map_bc_gpu);
        break;
    default:
        break;
    }
}

__global__
void gpu_build_boundary_conditions_lid_driven_cavity(NodeTypeMap* const map_bc_gpu)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    map_bc_gpu[index_scalar(x, y)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_NULL); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
    map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    map_bc_gpu[index_scalar(x, y)].set_rho_idx(0); // manually assigned (index of rho=RHO_0)

    if (x == 0 && y == 0)                           // SW
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTHWEST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == (N_X - 1) && y == 0)              // SE
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTHEAST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == (N_X - 1) && y == (N_Y - 1))      // NE
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTHEAST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(1); // manually assigned (index of ux=U_MAX)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == 0 && y == (N_Y - 1))              // NW
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTHWEST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(1); // manually assigned (index of ux=U_MAX)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (y == 0)                                // S
    {
        //map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (y == (N_Y - 1))                        // N
    {
        //map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_BOUNCE_BACK);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(1); // manually assigned (index of ux=U_MAX)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == 0)                                // W
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(WEST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == (N_X - 1))                        // E
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(EAST);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
}


__global__
void gpu_build_boundary_conditions_parallel_plates(NodeTypeMap* const map_bc_gpu)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    size_t idx_scalar = index_scalar(x, y);

    map_bc_gpu[index_scalar(x, y)].set_is_used(true); //set all nodes fluid inicially and no bc
    map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_NULL); //set all nodes fluid inicially and no bc

    if (x == 0 && y == 0)                           // SW
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == (N_X - 1) && y == 0)              // SE
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == (N_X - 1) && y == (N_Y - 1))      // NE
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == 0 && y == (N_Y - 1))              // NW
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (y == 0)                                // S
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(SOUTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (y == (N_Y - 1))                        // N
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_VEL_ZOUHE);
        map_bc_gpu[index_scalar(x, y)].set_direction(NORTH);
        map_bc_gpu[index_scalar(x, y)].set_ux_idx(0); // manually assigned (index of ux=0)
        map_bc_gpu[index_scalar(x, y)].set_uy_idx(0); // manually assigned (index of uy=0)
    }
    else if (x == 0)                                // W
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar(x, y)].set_direction(WEST); 
        map_bc_gpu[index_scalar(x, y)].set_rho_idx(1); // manually assigned (index of rho=RHO_IN)
    }
    else if (x == (N_X - 1))                        // E
    {
        map_bc_gpu[index_scalar(x, y)].set_BC_scheme(BC_SCHEME_PRES_NEBB);
        map_bc_gpu[index_scalar(x, y)].set_direction(EAST); 
        map_bc_gpu[index_scalar(x, y)].set_rho_idx(2); // manually assigned (index of rho=RHO_OUT)
    }
}


__device__
void gpu_boundary_conditions(NodeTypeMap * const nt_gpu, dfloat * f, const short unsigned int x, const short unsigned int y)
{
    /*
    IS_USED
        -> BC_SCHEME
            -> DIRECTION
    */
    if (nt_gpu->get_is_used())
    {
        switch (nt_gpu->get_BC_scheme())
        {
        case BC_NULL:
            return;
        case BC_SCHEME_VEL_ZOUHE:
            switch (nt_gpu->get_direction())
            {
            case NORTH:
                gpu_bc_vel_zouhe_N(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case SOUTH:
                gpu_bc_vel_zouhe_S(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case WEST:
                gpu_bc_vel_zouhe_W(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case EAST:
                gpu_bc_vel_zouhe_E(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case NORTHEAST:
                gpu_bc_vel_zouhe_NE(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case NORTHWEST:
                gpu_bc_vel_zouhe_NW(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case SOUTHEAST:
                gpu_bc_vel_zouhe_SE(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case SOUTHWEST:
                gpu_bc_vel_zouhe_SW(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            default:
                break;
            }
            break;

        case BC_SCHEME_VEL_BOUNCE_BACK:
            switch (nt_gpu->get_direction())
            {
            case NORTH:
                gpu_bc_vel_bounce_back_N(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case SOUTH:
                gpu_bc_vel_bounce_back_S(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case WEST:
                gpu_bc_vel_bounce_back_W(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case EAST:
                gpu_bc_vel_bounce_back_E(f, x, y, u_x_bc[nt_gpu->get_ux_idx()], u_y_bc[nt_gpu->get_uy_idx()]);
                break;
            case NORTHEAST:
                //gpu_bc_bounce_back_NE(f, x, y);
                break;
            case NORTHWEST:
                //gpu_bc_bounce_back_NW(f, x, y);
                break;
            case SOUTHEAST:
                //gpu_bc_bounce_back_SE(f, x, y);
                break;
            case SOUTHWEST:
                //gpu_bc_bounce_back_SW(f, x, y);
                break;
            default:
                break;
            }
            break;

        case BC_SCHEME_PRES_NEBB:
            switch (nt_gpu->get_direction())
            {
            case NORTH:
                gpu_bc_press_nebb_N(f, x, y, rho_bc[nt_gpu->get_rho_idx()]);
                break;
            case SOUTH:
                gpu_bc_press_nebb_S(f, x, y, rho_bc[nt_gpu->get_rho_idx()]);
                break;
            case WEST:
                gpu_bc_press_nebb_W(f, x, y, rho_bc[nt_gpu->get_rho_idx()]);
                break;
            case EAST:
                gpu_bc_press_nebb_E(f, x, y, rho_bc[nt_gpu->get_rho_idx()]);
                break;
            case NORTHEAST:
                break;
            case NORTHWEST:
                break;
            case SOUTHEAST:
                break;
            case SOUTHWEST:
                break;
            default:
                break;
            }
            break;

        case BC_SCHEME_BOUNCE_BACK:
            switch (nt_gpu->get_direction())
            {
            case NORTH:
                gpu_bc_bounce_back_N(f, x, y);
                break;
            case SOUTH:
                gpu_bc_bounce_back_S(f, x, y);
                break;
            case WEST:
                gpu_bc_bounce_back_W(f, x, y);
                break;
            case EAST:
                gpu_bc_bounce_back_E(f, x, y);
                break;
            case NORTHEAST:
                gpu_bc_bounce_back_NE(f, x, y);
                break;
            case NORTHWEST:
                gpu_bc_bounce_back_NW(f, x, y);
                break;
            case SOUTHEAST:
                gpu_bc_bounce_back_SE(f, x, y);
                break;
            case SOUTHWEST:
                gpu_bc_bounce_back_SW(f, x, y);
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
    }
}


__device__
void gpu_bc_vel_zouhe_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = 1 / (1 + (uy_w)) * (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 3)] +
        2 * (f[index_pop(x, y, 2)] + f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)]));

    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)] - 2.0 / 3 * rho_w * (uy_w);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] + 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) - 1.0 / 6 * rho_w * (uy_w) - 1.0 / 2 * rho_w * (ux_w);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] - 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) - 1.0 / 6 * rho_w * (uy_w) + 1.0 / 2 * rho_w * (ux_w);
}


__device__ 
void gpu_bc_vel_zouhe_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = 1 / (1 - (uy_w)) * (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 3)] +
        2 * (f[index_pop(x, y, 4)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)]));

    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)] + 2.0 / 3 * rho_w * (uy_w);
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) + 1.0 / 6 * rho_w * (uy_w) + 1.0 / 2 * rho_w * (ux_w);
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] + 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) + 1.0 / 6 * rho_w * (uy_w) - 1.0 / 2 * rho_w * (ux_w);    
}


__device__
void gpu_bc_vel_zouhe_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = 1 / (1 - (ux_w)) * (f[index_pop(x, y, 0)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 4)] +
        2 * (f[index_pop(x, y, 3)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)]));

    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)] + 2.0 / 3 * rho_w * (ux_w);
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) + 1.0 / 6 * rho_w * (ux_w) + 1.0 / 2 * rho_w * (uy_w);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] + 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) + 1.0 / 6 * rho_w * (ux_w) - 1.0 / 2 * rho_w * (uy_w);
}


__device__
void gpu_bc_vel_zouhe_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = 1 / (1 - (ux_w)) * (f[index_pop(x, y, 0)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 4)] +
        2 * (f[index_pop(x, y, 1)] + f[index_pop(x, y, 5)] + f[index_pop(x, y, 8)]));

    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)] - 2.0 / 3 * rho_w * (ux_w);
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] - 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) - 1.0 / 6 * rho_w * (ux_w)+1.0 / 2 * rho_w * (uy_w);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] + 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) - 1.0 / 6 * rho_w * (ux_w)-1.0 / 2 * rho_w * (uy_w);
}


__device__
void gpu_bc_vel_zouhe_NE(dfloat * f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + 2 * (f[index_pop(x, y, 1)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 5)])) / (1 + 5.0 / 6 * ux_w + 5.0 / 6 * uy_w);

    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)] + 2.0 / 3 * rho_w * (-ux_w);
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)] + 2.0 / 3 * rho_w * (-uy_w);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] + 1.0 / 6 * rho_w * (-ux_w - uy_w);

    //f[index_pop(x, y, 6)] = 1.0 / 12 * (-ux_w + uy_w);
    //f[index_pop(x, y, 8)] = 1.0 / 12 * (-ux_w + uy_w);
}


__device__
void gpu_bc_vel_zouhe_NW(dfloat * f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + 2 * (f[index_pop(x, y, 2)] + f[index_pop(x, y, 3)] + f[index_pop(x, y, 6)])) / (1 - 5.0 / 6 * ux_w + 5.0 / 6 * uy_w);

    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)] + 2.0 / 3 * rho_w * ux_w;
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)] + 2.0 / 3 * rho_w * (-uy_w);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] + 1.0 / 6 * rho_w * (ux_w - uy_w);

    //f[index_pop(x, y, 5)] = 1.0 / 12 * (ux_w + uy_w);
    //f[index_pop(x, y, 7)] = 1.0 / 12 * (ux_w + uy_w);
}


__device__
void gpu_bc_vel_zouhe_SE(dfloat * f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + 2 * (f[index_pop(x, y, 1)] + f[index_pop(x, y, 4)] + f[index_pop(x, y, 8)])) / (1 + 5.0 / 6 * ux_w - 5.0 / 6 * uy_w);
    
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)] + 2.0 / 3 * rho_w * uy_w;
    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)] + 2.0 / 3 * rho_w * (-ux_w);
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] + 1.0 / 6 * rho_w * (-ux_w + uy_w);

    //f[index_pop(x, y, 5)] = 1.0 / 12 * (-ux_w - uy_w);
    //f[index_pop(x, y, 7)] = 1.0 / 12 * (-ux_w - uy_w);

}


__device__
void gpu_bc_vel_zouhe_SW(dfloat * f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + 2 * (f[index_pop(x, y, 3)] + f[index_pop(x, y, 4)] + f[index_pop(x, y, 7)])) / (1 - 5.0 / 6 * ux_w - 5.0 / 6 * uy_w);
    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)] + 2.0 / 3 * rho_w * ux_w;
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)] + 2.0 / 3 * rho_w * uy_w;
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] + 1.0 / 6 * rho_w * (ux_w + uy_w);

    //f[index_pop(x, y, 6)] = 1.0 / 12 * (ux_w - uy_w);
    //f[index_pop(x, y, 8)] = 1.0 / 12 * (ux_w - uy_w);
}


__device__
void gpu_bc_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)];
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)];
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)];
}


__device__
void gpu_bc_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)];
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)];
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)];
}


__device__
void gpu_bc_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)];
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)];
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)];
}


__device__
void gpu_bc_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)];
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)];
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)];
}


__device__ 
void gpu_bc_vel_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 3)] + f[index_pop(x, y, 4)] +
        f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)]);

    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)] - 6 * W_1 * rho_w * (uy_w);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] - 6 * W_2 * rho_w * (ux_w + uy_w);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] - 6 * W_2 * rho_w * (-ux_w + uy_w);

}


__device__
void gpu_bc_vel_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 3)] + f[index_pop(x, y, 4)] +
        f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)]);

    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)] - 6 * W_1 * rho_w * (-uy_w);;
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 6 * W_2 * rho_w * (-ux_w - uy_w);
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] - 6 * W_2 * rho_w * (+ux_w - uy_w);
}


__device__
void gpu_bc_vel_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 3)] + f[index_pop(x, y, 4)] +
        f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)]);

    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)] - 6 * W_1 * rho_w * (-ux_w);
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 6 * W_2 * rho_w * (-ux_w - uy_w);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] - 6 * W_2 * rho_w * (-ux_w + uy_w);
}


__device__
void gpu_bc_vel_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w)
{
    const dfloat rho_w = (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 3)] + f[index_pop(x, y, 4)] +
        f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)]);

    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)] - 6 * W_1 * rho_w * (ux_w);;
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] - 6 * W_2 * rho_w * (ux_w - uy_w);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] - 6 * W_2 * rho_w * (ux_w + uy_w);
}


__device__
void gpu_bc_bounce_back_NE(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)];
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)];
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)];
}


__device__
void gpu_bc_bounce_back_NW(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)];
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)];
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)];
}


__device__
void gpu_bc_bounce_back_SE(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)];
    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)];
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)];
}


__device__
void gpu_bc_bounce_back_SW(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)];
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)];
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)];
}


__device__
void gpu_bc_press_nebb_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w)
{
    const dfloat u_y = -1 + (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 3)] +
        2 * (f[index_pop(x, y, 2)] + f[index_pop(x, y, 5)] + f[index_pop(x, y, 6)])) / rho_w;

    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)] - 2.0 / 3 * rho_w * (u_y);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] + 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) - 1.0 / 6 * rho_w * (u_y);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] - 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) - 1.0 / 6 * rho_w * (u_y);
}


__device__
void gpu_bc_press_nebb_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w)
{
    const dfloat u_y = 1 - (f[index_pop(x, y, 0)] + f[index_pop(x, y, 1)] + f[index_pop(x, y, 3)] +
        2 * (f[index_pop(x, y, 4)] + f[index_pop(x, y, 7)] + f[index_pop(x, y, 8)])) / rho_w;

    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)] + 2.0 / 3 * rho_w * (u_y);
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) + 1.0 / 6 * rho_w * (u_y);
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] + 1.0 / 2 * (f[index_pop(x, y, 1)] - f[index_pop(x, y, 3)]) + 1.0 / 6 * rho_w * (u_y);
}


__device__
void gpu_bc_press_nebb_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w)
{
    const dfloat u_x = 1 - (f[index_pop(x, y, 0)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 4)] +
        2 * (f[index_pop(x, y, 3)] + f[index_pop(x, y, 6)] + f[index_pop(x, y, 7)])) / rho_w;

    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)] + 2.0 / 3 * rho_w * (u_x);
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 7)] - 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) + 1.0 / 6 * rho_w * (u_x);
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 6)] + 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) + 1.0 / 6 * rho_w * (u_x);

}


__device__
void gpu_bc_press_nebb_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w)
{
    const dfloat u_x = -1 + (f[index_pop(x, y, 0)] + f[index_pop(x, y, 2)] + f[index_pop(x, y, 4)] + 
        2 * (f[index_pop(x, y, 1)] + f[index_pop(x, y, 5)] + f[index_pop(x, y, 8)])) / rho_w;

    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)] - 2.0 / 3 * rho_w * u_x;
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 8)] - 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) - 1.0 / 6 * rho_w * (u_x);
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 5)] + 1.0 / 2 * (f[index_pop(x, y, 2)] - f[index_pop(x, y, 4)]) - 1.0 / 6 * rho_w * (u_x);

}


__device__
void gpu_bc_free_slip_N(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 4)] = f[index_pop(x, y, 2)];
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 6)];
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 5)];
}


__device__
void gpu_bc_free_slip_S(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 2)] = f[index_pop(x, y, 4)];
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 8)];
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 7)];
}


__device__
void gpu_bc_free_slip_W(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 1)] = f[index_pop(x, y, 3)];
    f[index_pop(x, y, 5)] = f[index_pop(x, y, 6)];
    f[index_pop(x, y, 8)] = f[index_pop(x, y, 7)];
}


__device__
void gpu_bc_free_slip_E(dfloat* f, const short unsigned int x, const short unsigned int y)
{
    f[index_pop(x, y, 3)] = f[index_pop(x, y, 1)];
    f[index_pop(x, y, 6)] = f[index_pop(x, y, 5)];
    f[index_pop(x, y, 7)] = f[index_pop(x, y, 8)];
}
