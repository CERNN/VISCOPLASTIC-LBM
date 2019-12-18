/*
*   @file nodeTypeMap.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct to map boundary conditions 
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __NODE_TYPE_MAP_H
#define __NODE_TYPE_MAP_H

#include <builtin_types.h>
#include <stdint.h>

// OFFSET DEFINES
#define SPC_INTERP_BB_OFFSET 23
#define SAVE_POST_COL_OFFSET 22
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

// SAVE POST COLLISION DEFINE
#define SAVE_POST_COL (0b1 << SAVE_POST_COL_OFFSET)

// BC SCHEME DEFINES
#define BC_SCHEME_BITS (0b111 << BC_SCHEME_OFFSET)
#define BC_NULL (0b000)
#define BC_SCHEME_VEL_ZOUHE (0b001)
#define BC_SCHEME_VEL_BOUNCE_BACK (0b010)
#define BC_SCHEME_PRES_ZOUHE (0b011)
#define BC_SCHEME_FREE_SLIP (0b100)
#define BC_SCHEME_BOUNCE_BACK (0b101)
#define BC_SCHEME_INTERP_BOUNCE_BACK (0b110)
#define BC_SCHEME_SPECIAL (0b111)

// DIRECTION DEFINES
#define DIRECTION_BITS (0b11111 << DIRECTION_OFFSET)
#define NORTH (0b00000) //y=NY
#define SOUTH (0b00001) //y=0
#define WEST (0b00010)  //x=0
#define EAST (0b00011)  //x=NX
#define FRONT (0b00100) //z=NZ
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

// INTERPOLATED BOUNCE BACK SPECIAL DEFINES
#define SPC_INTERP_BB_BITS (0b11111111 << SPC_INTERP_BB_OFFSET)
#define UNKNOWN_POP_1 (0b00000001) // [x, y] = ( 1,  0)
#define UNKNOWN_POP_2 (0b00000010) // [x, y] = ( 0,  1)
#define UNKNOWN_POP_3 (0b00000100) // [x, y] = (-1,  0)
#define UNKNOWN_POP_4 (0b00001000) // [x, y] = ( 0, -1)
#define UNKNOWN_POP_5 (0b00010000) // [x, y] = ( 1,  1)
#define UNKNOWN_POP_6 (0b00100000) // [x, y] = (-1,  1)
#define UNKNOWN_POP_7 (0b01000000) // [x, y] = (-1, -1)
#define UNKNOWN_POP_8 (0b10000000) // [x, y] = ( 1, -1)

/*
*   Struct for mapping the type of each node using 32-bit variable for 
*   each node. The struct is organized as:
*   USED (1b) - SAVE_POST_COL (1b) - BC SCHEME (3b) - DIRECTION (5b) 
*   - GEOMETRY (1b) - UX_VAL_IDX (3b) - UY_VAL_IDX (3b) - UZ_VAL_IDX (3b) 
*   - RHO_VAL_IDX (3b) - SPC_INTERP_BB_BITS (8b)
*
*   With SPC_INTERP_BB_BITS being the MSB and RHO_VAL_IDX[0] the LSB. 
*   The bit sets meaning are explained below:
*
*   USED: node is used
*   SAVE_POST_COL: save post collision populations or not
*   BC SCHEME: scheme of boundary condition (null, Zou-He, bounce-back, etc.)
*   DIRECTION: normal direction of the node (N, S, W, E, F, B and possible 
*              combinations)
*   GEOMETRY: whether the node is concave or convex
*   UX_VAL_IDX: index for global array with the ux value for the node
*   UY_VAL_IDX: index for global array with the uy value for the node
*   UZ_VAL_IDX: index for global array with the uz value for the node
*   RHO_VAL_IDX: index for global array with the rho value for the node
*   SCP_INTERP_BC_BITS: bits to represent the known populations for the in
*       the direction bounce back interpolated boundary condition normal 
*
*/
typedef struct nodeTypeMap {
    uint32_t map;

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
    void setIsUsed(const bool isUsed)
    {
        if (isUsed)
            map |= (0x1 << IS_USED_OFFSET);
        else
            map &= ~(0x1 << IS_USED_OFFSET);
    }

    __device__ __host__
    bool getSavePostCol()
    {
        return (((map & SAVE_POST_COL) >> SAVE_POST_COL_OFFSET) 
            || (getSchemeBC() != BC_NULL));
    }

    __device__ __host__
    void setSavePostCol(const bool savePostCol)
    {
        if (savePostCol)
            map |= (0x1 << SAVE_POST_COL_OFFSET);
        else
            map &= ~(0x1 << SAVE_POST_COL_OFFSET);
    }
    
    __device__ __host__
    bool getIsUsed()
    {
        return ((map & IS_USED) >> IS_USED_OFFSET);
    }

    __device__ __host__
    void setSchemeBC(const char bcScheme)
    {
        if (bcScheme <= (BC_SCHEME_BITS >> BC_SCHEME_OFFSET))
            map = (map & ~BC_SCHEME_BITS) | (bcScheme << BC_SCHEME_OFFSET);
    }

    __device__ __host__
    char getSchemeBC()
    {
        return ((map & BC_SCHEME_BITS) >> BC_SCHEME_OFFSET);
    }

    __device__ __host__
    void setDirection(const char dir)
    {
        if (dir <= (DIRECTION_BITS >> DIRECTION_OFFSET))
            map = (map & ~DIRECTION_BITS) | (dir << DIRECTION_OFFSET);
    }

    __device__ __host__
    char getDirection()
    {
        return ((map & DIRECTION_BITS) >> DIRECTION_OFFSET);
    }

    __device__ __host__
    void setGeometry(const char geo)
    {
        if (geo <= (GEOMETRY_BITS >> GEOMETRY_OFFSET))
            map = (map & ~GEOMETRY_BITS) | (geo << GEOMETRY_OFFSET);
    }

    __device__ __host__
    char getGeometry()
    {
        return ((map & GEOMETRY_BITS) >> GEOMETRY_OFFSET);
    }

    __device__ __host__
    void setUxIdx(const char idx)
    {
        if (idx <= (UX_IDX_BITS >> UX_IDX_OFFSET))
            map = (map & ~UX_IDX_BITS) | (idx << UX_IDX_OFFSET);
    }

    __device__ __host__
    char getUxIdx()
    {
        return  ((map & UX_IDX_BITS) >> UX_IDX_OFFSET);
    }

    __device__ __host__
    void setUyIdx(const char idx)
    {
        if (idx <= (UY_IDX_BITS >> UY_IDX_OFFSET))
            map = (map & ~UY_IDX_BITS) | (idx << UY_IDX_OFFSET);
    }

    __device__ __host__
    char getUyIdx()
    {
        return ((map & UY_IDX_BITS) >> UY_IDX_OFFSET);
    }

    __device__ __host__
    void setUzIdx(const char idx)
    {
        if (idx <= (UZ_IDX_BITS >> UZ_IDX_OFFSET))
            map = (map & ~UZ_IDX_BITS) | (idx << UZ_IDX_OFFSET);
    }

    __device__ __host__
    char getUzIdx()
    {
        return ((map & UZ_IDX_BITS) >> UZ_IDX_OFFSET);
    }

    __device__ __host__
    void setRhoIdx(const char idx)
    {
        if (idx <= (RHO_IDX_BITS >> RHO_IDX_OFFSET))
            map = (map & ~RHO_IDX_BITS) | (idx << RHO_IDX_OFFSET);
    }

    __device__ __host__
    char getRhoIdx()
    {
        return ((map & RHO_IDX_BITS) >> RHO_IDX_OFFSET);
    }

    __device__ __host__
    char isBCLocal()
    {
        // if it's not free slip nor special, is local
        return !((this->getSchemeBC() == BC_SCHEME_FREE_SLIP) || 
            (this->getSchemeBC() == BC_SCHEME_SPECIAL));
    }

    __device__ __host__
    void setBitsUnknownPopsInterpBB(const char bits)
    {
        if (bits <= (SPC_INTERP_BB_BITS >> SPC_INTERP_BB_OFFSET))
            map = (bits & ~SPC_INTERP_BB_BITS) | (bits << SPC_INTERP_BB_OFFSET);
    }
    
    __device__ __host__
    void setUnknowPopInterpBB(const char pop)
    {
        // number of populations, excluding 0 (pop 1 is 0, pop 2 is 1, etc.)
        // D2Q9 scheme used (velocities described in UNKNOWN_POP defines)
        if (pop < 8)
            map |= ((0b1 << pop) << SPC_INTERP_BB_OFFSET);
    }

    __device__ __host__
    char getBitsUnknownPopsInterpBB()
    {
        return ((map & SPC_INTERP_BB_BITS) >> SPC_INTERP_BB_OFFSET);
    }

} NodeTypeMap;

#endif // !__NODE_TYPE_MAP_H
