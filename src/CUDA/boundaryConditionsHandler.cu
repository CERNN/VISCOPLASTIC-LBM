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
void gpuBoundaryConditions(NodeTypeMap* gpuNT, dfloat * f, dfloat* fNode, const short unsigned int x, const short unsigned int y, const short unsigned int z)
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
        gpuSchBounceBack(gpuNT, fNode);
        break;
    case BC_SCHEME_FREE_SLIP:
        gpuSchFreeSlip(gpuNT, f, fNode, x, y, z);
        break;
    case BC_SCHEME_VEL_BOUNCE_BACK:
        gpuSchVelBounceBack(gpuNT, fNode);
        break;  
    case BC_SCHEME_VEL_ZOUHE:
        gpuSchVelZouHe(gpuNT, fNode);
            break;
    case BC_SCHEME_PRES_ZOUHE:
        gpuSchPresZouHe(gpuNT, fNode);
    case BC_SCHEME_SPECIAL:
        gpuSchSpecial(gpuNT, f, fNode, x, y, z);
    default:
        break;
    }
}

__device__
void gpuSchBounceBack(NodeTypeMap* gpuNT, dfloat* fNode)
{
    switch(gpuNT->getDirection())
    {
    case NORTH:
        gpuBCBounceBackN(fNode);
        break;

    case SOUTH:
        gpuBCBounceBackS(fNode);
        break;

    case WEST:
        gpuBCBounceBackW(fNode);
        break;

    case EAST:
        gpuBCBounceBackE(fNode);
        break;

    case FRONT:
        gpuBCBounceBackF(fNode);
        break;

    case BACK:
        gpuBCBounceBackB(fNode);
        break;

    case NORTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNW(fNode);
        break;

    case NORTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNE(fNode);
        break;

    case NORTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNF(fNode);
        break;

    case NORTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNB(fNode);
        break;

    case SOUTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSW(fNode);
        break;

    case SOUTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSE(fNode);
        break;

    case SOUTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSF(fNode);
        break;

    case SOUTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSB(fNode);
        break;

    case WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWF(fNode);
        break;

    case WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWB(fNode);
        break;

    case EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEF(fNode);
        break;

    case EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEB(fNode);
        break;

    case NORTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWF(fNode);
        break;

    case NORTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWB(fNode);
        break;

    case NORTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEF(fNode);
        break;

    case NORTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEB(fNode);
        break;

    case SOUTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWF(fNode);
        break;

    case SOUTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWB(fNode);
        break;

    case SOUTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEF(fNode);
        break;

    case SOUTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEB(fNode);
        break;

    default:
        break;
    }
}


__device__
void gpuSchVelBounceBack(NodeTypeMap* gpuNT, dfloat* fNode)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelBounceBackN(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelBounceBackS(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelBounceBackW(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelBounceBackE(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelBounceBackF(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelBounceBackB(fNode, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    default:
        break;
    }
#endif
}


__device__
void gpuSchFreeSlip(NodeTypeMap* gpuNT, dfloat* f, dfloat* fNode, const short unsigned int x, const short unsigned int y, const short unsigned int z)
{
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCFreeSlipN(f, fNode, x, y, z);
        break;

    case SOUTH:
        gpuBCFreeSlipS(f, fNode, x, y, z);
        break;

    case WEST:
        gpuBCFreeSlipW(f, fNode, x, y, z);
        break;

    case EAST:
        gpuBCFreeSlipE(f, fNode, x, y, z);
        break;

    case FRONT:
        gpuBCFreeSlipF(f, fNode, x, y, z);
        break;

    case BACK:
        gpuBCFreeSlipB(f, fNode, x, y, z);
        break;
    default:
        break;
    }
}


__device__
void gpuSchPresZouHe(NodeTypeMap* gpuNT, dfloat* fNode)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCPresZouHeN(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case SOUTH:
        gpuBCPresZouHeS(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case WEST:
        gpuBCPresZouHeW(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case EAST:
        gpuBCPresZouHeE(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case FRONT:
        gpuBCPresZouHeF(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case BACK:
        gpuBCPresZouHeB(fNode, rhoBC[gpuNT->getRhoIdx()]);
        break;
    default:
        break;
    }
#endif
}


__device__
void gpuSchVelZouHe(NodeTypeMap * gpuNT, dfloat * fNode)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelZouHeN(fNode, uxBC[gpuNT->getUxIdx()], 
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelZouHeS(fNode, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelZouHeW(fNode, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelZouHeE(fNode, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelZouHeF(fNode, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelZouHeB(fNode, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;
    default:
        break;
    }
#endif
}