#ifndef STRUCTS_CUH
#define STRUCTS_CUH

#include <builtin_types.h>
#include <stdint.h>

// OFFSET DEFINES
#define IS_USED_OFFSET 21
#define BC_SCHEME_OFFSET 18
#define DIRECTION_OFFSET 13
#define GEOMETRY_OFFSET 12
#define UX_IDX_OFFSET 9
#define UY_IDX_OFFSET 6
#define UZ_IDX_OFFSET 3
#define RHO_IDX_OFFSET 0

// USED DEFINE
#define IS_USED (0b1 << IS_USED_OFFSET)

// BC SCHEME DEFINES
#define BC_SCHEME_BITS (0b111 << BC_SCHEME_OFFSET)
#define BC_NULL (0b000)
#define BC_SCHEME_VEL_NEBB (0b001)
#define BC_SCHEME_VEL_BOUNCE_BACK (0b010)
#define BC_SCHEME_PRES_NEBB (0b011)
#define BC_SCHEME_FREE_SLIP (0b100)
#define BC_SCHEME_BOUNCE_BACK (0b101)
#define BC_SCHEME_PRES_ANTI_BB (0b110)

// DIRECTION DEFINES
#define DIRECTION_BITS (0b11111 << DIRECTION_OFFSET)
#define NORTH (0b00000) //y=N_Y
#define SOUTH (0b00001) //y=0
#define WEST (0b00010)  //x=0
#define EAST (0b00011)  //x=N_X
#define FRONT (0b00100) //z=N_Z
#define BACK (0b00101)  //z=0
#define NORTH_WEST (0b00110)
#define NORTH_EAST (0b00111)
#define NORTH_FRONT (0b01000)
#define NORTH_BACK (0b01001)
#define SOUTH_WEST (0b01010)
#define SOUTH_EAST (0b01011)
#define SOUTH_FRONT (0b01100)
#define SOUTH_BACK (0b01101)
#define WEST_FRONT (0b01110)
#define WEST_BACK (0b01111)
#define EAST_FRONT (0b10000)
#define EAST_BACK (0b10001)
#define NORTH_WEST_FRONT (0b10010)
#define NORTH_WEST_BACK (0b10011)
#define NORTH_EAST_FRONT (0b10100)
#define NORTH_EAST_BACK (0b10101)
#define SOUTH_WEST_FRONT (0b10110)
#define SOUTH_WEST_BACK (0b10111)
#define SOUTH_EAST_FRONT (0b11000)
#define SOUTH_EAST_BACK (0b11001)

// NODE GEOMETRY
#define GEOMETRY_BITS (0b1 << GEOMETRY_OFFSET)
#define CONCAVE (0b0)
#define CONVEX (0b1)

// INDEXES DEFINES
#define UX_IDX_BITS (0b111 << UX_IDX_OFFSET)
#define UY_IDX_BITS (0b111 << UY_IDX_OFFSET)
#define UZ_IDX_BITS (0b111 << UZ_IDX_OFFSET)
#define RHO_IDX_BITS (0b111 << RHO_IDX_OFFSET)


/*
*   Struct for mapping the type of each node using 32-bit variable for each node
*   The struct is organized as:
*   USED (1b) - BC SCHEME (3b) - DIRECTION (5b) - GEOMETRY (1b)- UX_VAL_IDX (3b) - UY_VAL_IDX (3b) -
*   UZ_VAL_IDX (3b) - RHO_VAL_IDX (3b)
*
*   With USED being the MSB and RHO_VAL_IDX the LSB. The bit sets meaning are explained below:
*
*   USED: node is used
*   BC SCHEME: scheme of boundary condition (null, Zou-He, bounce-back, etc.)
*   DIRECTION: normal direction of the node (N, S, W, E, F, B and possible combinations)
*   GEOMETRY: whether the node is concave or convex
*   UX_VAL_IDX: index for global array with the ux value for the node
*   UY_VAL_IDX: index for global array with the uy value for the node
*   UZ_VAL_IDX: index for global array with the uz value for the node
*   RHO_VAL_IDX: index for global array with the rho value for the node
*/
typedef struct nodeTypeMap {
    __int32 map;


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
            map |= (1 << IS_USED_OFFSET);
        else
            map &= ~IS_USED_OFFSET;
    }

    __device__ __host__
        bool get_is_used()
    {
        return ((map & IS_USED) >> IS_USED_OFFSET);
    }

    __device__ __host__
        void set_BC_scheme(const char bc_scheme)
    {
        if (bc_scheme <= (BC_SCHEME_BITS >> BC_SCHEME_OFFSET))
            map = (map & ~BC_SCHEME_BITS) | (bc_scheme << BC_SCHEME_OFFSET);
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
            map = (map & ~DIRECTION_BITS) | (dir << DIRECTION_OFFSET);
    }

    __device__ __host__
        char get_direction()
    {
        return ((map & DIRECTION_BITS) >> DIRECTION_OFFSET);
    }

    __device__ __host__
        void set_geometry(const char geo)
    {
        if (geo <= (GEOMETRY_BITS >> GEOMETRY_OFFSET))
            map = (map & ~GEOMETRY_BITS) | (geo << GEOMETRY_OFFSET);
    }

    __device__ __host__
        char get_geometry()
    {
        return ((map & GEOMETRY_BITS) >> GEOMETRY_OFFSET);
    }

    __device__ __host__
        void set_ux_idx(const char idx)
    {
        if (idx <= (UX_IDX_BITS >> UX_IDX_OFFSET))
            map = (map & ~UX_IDX_BITS) | (idx << UX_IDX_OFFSET);
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
            map = (map & ~UY_IDX_BITS) | (idx << UY_IDX_OFFSET);
    }

    __device__ __host__
        char get_uy_idx()
    {
        return ((map & UY_IDX_BITS) >> UY_IDX_OFFSET);
    }

    __device__ __host__
        void set_uz_idx(const char idx)
    {
        if (idx <= (UZ_IDX_BITS >> UZ_IDX_OFFSET))
            map = (map & ~UZ_IDX_BITS) | (idx << UZ_IDX_OFFSET);
    }

    __device__ __host__
        char get_uz_idx()
    {
        return ((map & UZ_IDX_BITS) >> UZ_IDX_OFFSET);
    }

    __device__ __host__
        void set_rho_idx(const char idx)
    {
        if (idx <= (RHO_IDX_BITS >> RHO_IDX_OFFSET))
            map = (map & ~RHO_IDX_BITS) | (idx << RHO_IDX_OFFSET);
    }

    __device__ __host__
        char get_rho_idx()
    {
        return ((map & RHO_IDX_BITS) >> RHO_IDX_OFFSET);
    }
}NodeTypeMap;

#endif // !STRUCTS_CUH
