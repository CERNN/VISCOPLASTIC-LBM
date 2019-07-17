#ifndef BOUNDARY_CONDITIONS_D2Q9_CUH
#define BOUNDARY_CONDITIONS_D2Q9_CUH

//TODO: ADD SUPPORT TO CONVEX AND CONCAVE NODES, RHO IN CORNERS FOR ZOU HE, 
//      ADD MORE SCHEMES

#include "./../common/func_idx.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// OFFSET DEFINES
#define IS_USED_OFFSET 15
#define BC_SCHEME_OFFSET 12
#define DIRECTION_OFFSET 9
#define UX_IDX_OFFSET 6
#define UY_IDX_OFFSET 3
#define RHO_IDX_OFFSET 0

// USED DEFINE
#define IS_USED (0b1 << IS_USED_OFFSET)

// BC SCHEME DEFINES
#define BC_SCHEME_BITS (0b111 << BC_SCHEME_OFFSET)
#define BC_NULL (0b000)
#define BC_SCHEME_VEL_ZOUHE (0b001)
#define BC_SCHEME_VEL_BOUNCE_BACK (0b010)
#define BC_SCHEME_PRES_NEBB (0b011)
#define BC_SCHEME_FREE_SLIP (0b100)
#define BC_SCHEME_BOUNCE_BACK (0b101)

// DIRECTION DEFINES
#define DIRECTION_BITS (0b111 << DIRECTION_OFFSET)
#define NORTH (0b000)
#define SOUTH (0b001)
#define WEST (0b010)
#define EAST (0b011)
#define NORTHEAST (0b100)
#define NORTHWEST (0b110)
#define SOUTHEAST (0b101)
#define SOUTHWEST (0b111)

// INDEXES DEFINES
#define UX_IDX_BITS (0b111 << UX_IDX_OFFSET)
#define UY_IDX_BITS (0b111 << UY_IDX_OFFSET)
#define RHO_IDX_BITS (0b111 << RHO_IDX_OFFSET)


/*
*   Struct for mapping the type of each node using 16-bit variable for each node
*   The struct is organized as:
*   USED (1b) - BC SCHEME (3b) - DIRECTION (3b) - UX_VAL_IDX (3b) - UY_VAL_IDX (3b) - RHO_VAL_IDX (3b)
*   With USED being the MSB. The bit sets meaning are explained below:
*   USED: node is used
*   BC SCHEME: scheme of boundary condition (Zou-He, bounce-back, null) 
*   DIRECTION: normal direction of the node (N, S, W, E, NE, NW, SE, SW)
*   UX_VAL_IDX: index for global array with the ux value for the node
*   UY_VAL_IDX: index for global array with the uy value for the node
*   RHO_VAL_IDX: index for global array with the rho value for the node
*/
typedef struct nodeTypeMap {
    __int16 map;


    __device__ __host__
    nodeTypeMap() //constructor
    {
        map = 0;
    }

    __device__ __host__
    ~nodeTypeMap() //destructor
    {
        map = 0;
    }

    __device__ __host__ 
    void set_is_used(const bool is_used)
    {
        if (is_used)
            map ^= IS_USED;
    }

    __device__ __host__ 
    bool get_is_used()
    {
        return (map & IS_USED);
    }

    __device__ __host__ 
    void set_BC_scheme(const char bc_scheme)
    {
        if (bc_scheme <= (BC_SCHEME_BITS >> BC_SCHEME_OFFSET))
            map += bc_scheme << BC_SCHEME_OFFSET;
    }
    
    __device__ __host__ 
    char get_BC_scheme()
    {
        return ((map & BC_SCHEME_BITS) >> BC_SCHEME_OFFSET);
    }

    __device__ __host__ 
    void set_direction(const char dir)
    {
        if (dir <= (DIRECTION_BITS >> DIRECTION_OFFSET))
            map += (dir << DIRECTION_OFFSET);
    }

    __device__ __host__ 
    char get_direction()
    {
        return ((map & DIRECTION_BITS) >> DIRECTION_OFFSET);
    }

    __device__ __host__ 
    void set_ux_idx(const char idx)
    {
        if (idx <= (UX_IDX_BITS >> UX_IDX_OFFSET))
            map += (idx << UX_IDX_OFFSET);
    }

    __device__ __host__ 
    char get_ux_idx()
    {
        return  ((map & UX_IDX_BITS) >> UX_IDX_OFFSET);
    }

    __device__ __host__ 
    void set_uy_idx(const char idx)
    {
        if (idx <= (UY_IDX_BITS >> UY_IDX_OFFSET))
            map += (idx << UY_IDX_OFFSET);
    }

    __device__ __host__ 
    char get_uy_idx()
    {
        return ((map & UY_IDX_BITS) >> UY_IDX_OFFSET);
    }

    __device__ __host__ 
    void set_rho_idx(const char idx)
    {
        if (idx <= (RHO_IDX_BITS >> RHO_IDX_OFFSET))
            map += (idx << RHO_IDX_OFFSET);
    }

    __device__ __host__ 
    char get_rho_idx()
    {
        return ((map & RHO_IDX_BITS) >> RHO_IDX_OFFSET);
    }
}NodeTypeMap;


/*
*   Builds boundary conditions map
*   \param map_bc_gpu: device pointer to the boundary conditions map
*   \param BC_TYPE: type of boundary condition (options in "var.h" defines)
*/
__host__
void build_boundary_conditions(NodeTypeMap* const map_bc_gpu, const int BC_TYPE);


/*
*   Builds boundary conditions map for lid driven cavity
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_lid_driven_cavity(NodeTypeMap* const map_bc_gpu);


/*
*   Builds boundary conditions map for parallel plates
*   \param map_bc_gpu: device pointer to the boundary conditions map
*/
__global__
void gpu_build_boundary_conditions_parallel_plates(NodeTypeMap* const map_bc_gpu);

__global__
void gpu_build_boundary_conditions_periodic_channel(NodeTypeMap* const map_bc_gpu);


/*
*   Applies boundary conditions given node type and its population
*   \param nt_gpu: node's map
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_boundary_conditions(NodeTypeMap* const nt_gpu, dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies Zou-He velocity boundary conditions on north wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on south wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on west wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on east wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on northeast corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_NE(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on northwest corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_NW(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on southeast corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_SE(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies Zou-He velocity boundary conditions on southwest corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_zouhe_SW(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies bounce back boundary conditions on north wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on south wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_S(dfloat * f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on west wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on east wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y);



/*
*   Applies bounce back boundary conditions on northeast corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_NE(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on northwest corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_NW(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on southeast corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_SE(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies bounce back boundary conditions on southwest corner node (concave)
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_bounce_back_SW(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies velocity bounce back boundary conditions on north wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_bounce_back_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies velocity bounce back boundary conditions on south wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_bounce_back_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies velocity bounce back boundary conditions on west wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's ux velocity
*   \param uy_w: node's ux velocity
*/
__device__
void gpu_bc_vel_bounce_back_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies velocity bounce back boundary conditions on east wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param ux_w: node's x velocity
*   \param uy_w: node's y velocity
*/
__device__
void gpu_bc_vel_bounce_back_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat ux_w, const dfloat uy_w);


/*
*   Applies pressure non equilibirum bounce back boundary conditions on north wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param rho_w: node's density
*/
__device__
void gpu_bc_press_nebb_N(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w);


/*
*   Applies pressure non equilibirum bounce back boundary conditions on south wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param rho_w: node's density
*/
__device__
void gpu_bc_press_nebb_S(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w);


/*
*   Applies pressure non equilibirum bounce back boundary conditions on west wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param rho_w: node's density
*/
__device__
void gpu_bc_press_nebb_W(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w);


/*
*   Applies pressure non equilibirum bounce back boundary conditions on east wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*   \param rho_w: node's density
*/
__device__
void gpu_bc_press_nebb_E(dfloat* f, const short unsigned int x, const short unsigned int y, const dfloat rho_w);


/*
*   Applies free slip boundary conditions on north wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_free_slip_N(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies free slip boundary conditions on south wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_free_slip_S(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies free slip boundary conditions on west wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_free_slip_W(dfloat* f, const short unsigned int x, const short unsigned int y);


/*
*   Applies free slip boundary conditions on east wall node
*   \param f[(N_X, N_Y, Q)]: grid of populations from 0 to 8
*   \param x: node's x value
*   \param y: node's y value
*/
__device__
void gpu_bc_free_slip_E(dfloat* f, const short unsigned int x, const short unsigned int y);


#endif // !BOUNDARY_CONDITIONS_D2Q9_CUH