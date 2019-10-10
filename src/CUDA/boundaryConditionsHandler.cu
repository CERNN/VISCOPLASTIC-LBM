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

#include "boundaryConditionsHandler.h"


__device__
void gpuBoundaryConditions(NodeTypeMap* gpuNT, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    /*
    -> BC_SCHEME
        -> DIRECTION
            -> GEOMETRY
    */
    switch(gpuNT->getSchemeBC())
    {
    case BC_NULL:
        return;
    case BC_SCHEME_BOUNCE_BACK:
        gpuSchBounceBack(gpuNT, f, x, y, z);
        break;
    case BC_SCHEME_FREE_SLIP:
        gpuSchFreeSlip(gpuNT, f, x, y, z);
        break;
    case BC_SCHEME_VEL_BOUNCE_BACK:
        gpuSchVelBounceBack(gpuNT, f, x, y, z);
        break;  
    case BC_SCHEME_VEL_ZOUHE:
        gpuSchVelZouHe(gpuNT, f, x, y, z);
            break;
    case BC_SCHEME_PRES_ZOUHE:
        gpuSchPresZouHe(gpuNT, f, x, y, z);
    case BC_SCHEME_SPECIAL:
        gpuSchSpecial(gpuNT, f, x, y, z);
    default:
        break;
    }
}

__device__
void gpuSchBounceBack(NodeTypeMap* gpuNT, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch(gpuNT->getDirection())
    {
    case NORTH:
        gpuBCBounceBackN(f, x, y, z);
        break;

    case SOUTH:
        gpuBCBounceBackS(f, x, y, z);
        break;

    case WEST:
        gpuBCBounceBackW(f, x, y, z);
        break;

    case EAST:
        gpuBCBounceBackE(f, x, y, z);
        break;

    case FRONT:
        gpuBCBounceBackF(f, x, y, z);
        break;

    case BACK:
        gpuBCBounceBackB(f, x, y, z);
        break;

    case NORTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNW(f, x, y, z);
        break;

    case NORTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNE(f, x, y, z);
        break;

    case NORTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNF(f, x, y, z);
        break;

    case NORTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNB(f, x, y, z);
        break;

    case SOUTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSW(f, x, y, z);
        break;

    case SOUTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSE(f, x, y, z);
        break;

    case SOUTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSF(f, x, y, z);
        break;

    case SOUTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSB(f, x, y, z);
        break;

    case WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWF(f, x, y, z);
        break;

    case WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWB(f, x, y, z);
        break;

    case EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEF(f, x, y, z);
        break;

    case EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEB(f, x, y, z);
        break;

    case NORTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWF(f, x, y, z);
        break;

    case NORTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWB(f, x, y, z);
        break;

    case NORTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEF(f, x, y, z);
        break;

    case NORTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEB(f, x, y, z);
        break;

    case SOUTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWF(f, x, y, z);
        break;

    case SOUTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWB(f, x, y, z);
        break;

    case SOUTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEF(f, x, y, z);
        break;

    case SOUTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEB(f, x, y, z);
        break;

    default:
        break;
    }
}


__device__
void gpuSchVelBounceBack(NodeTypeMap* gpuNT, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelBounceBackN(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelBounceBackS(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelBounceBackW(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelBounceBackE(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelBounceBackF(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelBounceBackB(f, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    default:
        break;
    }
#endif
}


__device__
void gpuSchFreeSlip(NodeTypeMap* gpuNT, dfloat* f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCFreeSlipN(f, x, y, z);
        break;

    case SOUTH:
        gpuBCFreeSlipS(f, x, y, z);
        break;

    case WEST:
        gpuBCFreeSlipW(f, x, y, z);
        break;

    case EAST:
        gpuBCFreeSlipE(f, x, y, z);
        break;

    case FRONT:
        gpuBCFreeSlipF(f, x, y, z);
        break;

    case BACK:
        gpuBCFreeSlipB(f, x, y, z);
        break;
        
    case NORTH_WEST:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipNW(f, x, y, z);
        break;

    case NORTH_EAST:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipNE(f, x, y, z);
        break;

    case NORTH_FRONT:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipNF(f, x, y, z);
        break;

    case NORTH_BACK:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipNB(f, x, y, z);
        break;

    case SOUTH_WEST:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipSW(f, x, y, z);
        break;

    case SOUTH_EAST:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipSE(f, x, y, z);
        break;

    case SOUTH_FRONT:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipSF(f, x, y, z);
        break;

    case SOUTH_BACK:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipSB(f, x, y, z);
        break;

    case WEST_FRONT:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipWF(f, x, y, z);
        break;

    case WEST_BACK:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipWB(f, x, y, z);
        break;

    case EAST_FRONT:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipEF(f, x, y, z);
        break;

    case EAST_BACK:
        if (gpuNT->getGeometry() == CONCAVE)
            gpuBCFreeSlipEB(f, x, y, z);
        break;
    default:
        break;
    }
}


__device__
void gpuSchPresZouHe(NodeTypeMap* gpuNT, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCPresZouHeN(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case SOUTH:
        gpuBCPresZouHeS(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case WEST:
        gpuBCPresZouHeW(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case EAST:
        gpuBCPresZouHeE(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case FRONT:
        gpuBCPresZouHeF(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case BACK:
        gpuBCPresZouHeB(f, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;
    default:
        break;
    }
#endif
}


__device__
void gpuSchVelZouHe(NodeTypeMap * gpuNT, dfloat * f, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelZouHeN(f, x, y, z, uxBC[gpuNT->getUxIdx()], 
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelZouHeS(f, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelZouHeW(f, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelZouHeE(f, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelZouHeF(f, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelZouHeB(f, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;
    default:
        break;
    }
#endif
}