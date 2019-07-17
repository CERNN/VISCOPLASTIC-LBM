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

#include "pch.h"
#include "lbm2.h"


void initialisation(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y)
{
    // all nodes initially are equal
    double f_ini[Q];
    for (unsigned int i = 0; i < Q; i++)
    {
        f_ini[i] = w[i]; // rho = 1, u_x = 0, u_y = 0
    }

    // initialize nodes
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            f_0[scalar_index2d(x, y)]  = f_ini[0];
            f[scalar_index_pop(x, y, 1)] = f_ini[1];
            f[scalar_index_pop(x, y, 2)] = f_ini[2];
            f[scalar_index_pop(x, y, 3)] = f_ini[3];
            f[scalar_index_pop(x, y, 4)] = f_ini[4];
            f[scalar_index_pop(x, y, 5)] = f_ini[5];
            f[scalar_index_pop(x, y, 6)] = f_ini[6];
            f[scalar_index_pop(x, y, 7)] = f_ini[7];
            f[scalar_index_pop(x, y, 8)] = f_ini[8];
            f_post[scalar_index_pop(x, y, 1)] = 0;
            f_post[scalar_index_pop(x, y, 2)] = 0;
            f_post[scalar_index_pop(x, y, 3)] = 0;
            f_post[scalar_index_pop(x, y, 4)] = 0;
            f_post[scalar_index_pop(x, y, 5)] = 0;
            f_post[scalar_index_pop(x, y, 6)] = 0;
            f_post[scalar_index_pop(x, y, 7)] = 0;
            f_post[scalar_index_pop(x, y, 8)] = 0;
            u_x[scalar_index2d(x, y)] = 0;
            u_y[scalar_index2d(x, y)] = 0;
            rho[scalar_index2d(x, y)] = 1;
        }
}


void collision(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y)
{
    double f_eq[Q];
    double T_OMEGA = 1 - OMEGA;
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            size_t index2d = scalar_index2d(x, y);
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            
            double u_u = u_x[index2d] * u_x[index2d] + u_y[index2d] * u_y[index2d];
            double rho_w0 = rho[index2d] * W_0;
            double rho_w1 = rho[index2d] * W_1;
            double rho_w2 = rho[index2d] * W_2;
        
            // calc for f_eq: 
            // f_eq[i] = rho * w[i] (1 + (u-c[i]) / (c_s)^2 + (u-c[i])^2 / (2 * (c_s)^2) + (u-u) / (2 * (c_s)^2))
            // a-b: dot product of a and b; w[i]: velocity weight;  c_s: sound velocity
            f_eq[0] = rho_w0 * (1 - u_u * 3 / 2.0);
            
            f_eq[1] = rho_w1 * (1 + u_x[index2d]                  * 3 + ((u_x[index2d]) * (u_x[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[2] = rho_w1 * (1 + u_y[index2d]                  * 3 + ((u_y[index2d]) * (u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[3] = rho_w1 * (1 - u_x[index2d]                  * 3 + ((u_x[index2d]) * (u_x[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[4] = rho_w1 * (1 - u_y[index2d]                  * 3 + ((u_y[index2d]) * (u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[5] = rho_w2 * (1 + (  u_x[index2d] + u_y[index2d]) * 3 + ((  u_x[index2d] + u_y[index2d]) * (  u_x[index2d] + u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[6] = rho_w2 * (1 + (- u_x[index2d] + u_y[index2d]) * 3 + ((- u_x[index2d] + u_y[index2d]) * (- u_x[index2d] + u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[7] = rho_w2 * (1 + (- u_x[index2d] - u_y[index2d]) * 3 + ((- u_x[index2d] - u_y[index2d]) * (- u_x[index2d] - u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[8] = rho_w2 * (1 + (  u_x[index2d] - u_y[index2d]) * 3 + ((  u_x[index2d] - u_y[index2d]) * (  u_x[index2d] - u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            

            // calc for f*:
            // f* = (1 - 1 / TAU) * f + (1 / TAU) * f_eq
            f_0[index2d]  = T_OMEGA * f_0[index2d]  + OMEGA * f_eq[0];
            f_post[index_pop_1] = T_OMEGA * f[index_pop_1] + OMEGA * f_eq[1];
            f_post[index_pop_1 + 1] = T_OMEGA * f[index_pop_1 + 1] + OMEGA * f_eq[2];
            f_post[index_pop_1 + 2] = T_OMEGA * f[index_pop_1 + 2] + OMEGA * f_eq[3];
            f_post[index_pop_1 + 3] = T_OMEGA * f[index_pop_1 + 3] + OMEGA * f_eq[4];
            f_post[index_pop_1 + 4] = T_OMEGA * f[index_pop_1 + 4] + OMEGA * f_eq[5];
            f_post[index_pop_1 + 5] = T_OMEGA * f[index_pop_1 + 5] + OMEGA * f_eq[6];
            f_post[index_pop_1 + 6] = T_OMEGA * f[index_pop_1 + 6] + OMEGA * f_eq[7];
            f_post[index_pop_1 + 7] = T_OMEGA * f[index_pop_1 + 7] + OMEGA * f_eq[8];
        }
}


void collision_generic(double * f, double * f_post, double * f_0, double * rho, double * u_x, double * u_y)
{
    double f_eq[Q];
    double T_OMEGA = 1 - OMEGA;
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            size_t index2d = scalar_index2d(x, y);
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            
            f_eq[0] = f_eq_generic(rho[index2d], u_x[index2d], u_y[index2d], 0);
            f_0[index2d] = T_OMEGA * f_0[index2d] + OMEGA * f_eq[0];
            for (int i = 1; i < Q; i++)
            {
                // calc for f_eq: 
                // f_eq[i] = rho * w[i] (1 + (u-c[i]) / (c_s)^2 + (u-c[i])^2 / (2 * (c_s)^2) + (u-u) / (2 * (c_s)^2))
                // a-b: dot product of a and b; w[i]: velocity weight;  c_s: sound velocity
                f_eq[i] = f_eq_generic(rho[index2d], u_x[index2d], u_y[index2d], i);

                // calc for f*:
                // f* = (1 - 1 / TAU) * f + (1 / TAU) * f_eq
                f_post[index_pop_1 + (i - 1)] = T_OMEGA * f[index_pop_1 + (i - 1)] + OMEGA * f_eq[i];
            }
        }
}


void collision_regularized(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y)
{
    double f_eq[Q];
    for (int x = 0; x < N_X; x++)
    {
        for(int y = 0; y < N_Y; y++)
        {
            size_t index2d = scalar_index2d(x, y);
            // -------- equilibrium populations evaluation ---------
            double u_u = u_x[index2d] * u_x[index2d] + u_y[index2d] * u_y[index2d];
            double rho_w0 = rho[index2d] * W_0;
            double rho_w1 = rho[index2d] * W_1;
            double rho_w2 = rho[index2d] * W_2;

            // calc for f_eq: 
            // f_eq[i] = rho * w[i] (1 + (u-c[i]) / (c_s)^2 + (u-c[i])^2 / (2 * (c_s)^2) + (u-u) / (2 * (c_s)^2))
            // a-b: dot product of a and b; w[i]: velocity weight;  c_s: sound velocity
            f_eq[0] = rho_w0 * (1 - u_u * C_S_INV_SQR / 2.0);

            f_eq[1] = rho_w1 * (1 + u_x[index2d] * 3 + ((u_x[index2d]) * (u_x[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[2] = rho_w1 * (1 + u_y[index2d] * 3 + ((u_y[index2d]) * (u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[3] = rho_w1 * (1 - u_x[index2d] * 3 + ((u_x[index2d]) * (u_x[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[4] = rho_w1 * (1 - u_y[index2d] * 3 + ((u_y[index2d]) * (u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[5] = rho_w2 * (1 + ( u_x[index2d] + u_y[index2d]) * 3 + (( u_x[index2d] + u_y[index2d]) * ( u_x[index2d] + u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[6] = rho_w2 * (1 + (-u_x[index2d] + u_y[index2d]) * 3 + ((-u_x[index2d] + u_y[index2d]) * (-u_x[index2d] + u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[7] = rho_w2 * (1 + (-u_x[index2d] - u_y[index2d]) * 3 + ((-u_x[index2d] - u_y[index2d]) * (-u_x[index2d] - u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            f_eq[8] = rho_w2 * (1 + ( u_x[index2d] - u_y[index2d]) * 3 + (( u_x[index2d] - u_y[index2d]) * ( u_x[index2d] - u_y[index2d]) * 9 / 2.0) - u_u * 3 / 2.0);
            // ------------------------------------------------------

            double f_1 = 0, pi_neq_xx = 0, pi_neq_yy = 0, pi_neq_xy = 0;

            for (int i = 1; i < Q; i++)
            {
                pi_neq_xx += c_x[i] * c_x[i] * (f[scalar_index_pop(x, y, i)] - f_eq[i]);
                pi_neq_yy += c_y[i] * c_y[i] * (f[scalar_index_pop(x, y, i)] - f_eq[i]);
                pi_neq_xy += c_x[i] * c_y[i] * (f[scalar_index_pop(x, y, i)] - f_eq[i]);
            }

            f_0[index2d] = f_eq[0]; // simplificatian due to c=(0,0)

            for (int i = 1; i < Q; i++)
            {
                f_1 = 4.5 * w[i] * ((c_x[i] * c_x[i] - 1.0 / 3) * pi_neq_xx + 
                    2 * (c_x[i] * c_y[i]) * pi_neq_xy + 
                    (c_y[i] * c_y[i] - 1.0 / 3) * pi_neq_yy);
                f_post[scalar_index_pop(x, y, i)] = f_eq[i] + (1.0 - OMEGA)*f_1;
            }
        }
    }
}


void streaming(double* f1, double* f2)
{
    // stream fluid nodes: 
    // f1[x][y][i] = f2[x - c[i]][y - c[i]][i]

    for (unsigned int x = 1; x < N_X - 1; x++)      // exclude borders
        for (unsigned int y = 1; y < N_Y - 1; y++)  // exclude borders
        {
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            f1[scalar_index_pop(x + 1, y    , 1)] = f2[index_pop_1];
            f1[scalar_index_pop(x    , y + 1, 2)] = f2[index_pop_1 + 1];
            f1[scalar_index_pop(x - 1, y    , 3)] = f2[index_pop_1 + 2];
            f1[scalar_index_pop(x    , y - 1, 4)] = f2[index_pop_1 + 3];
            f1[scalar_index_pop(x + 1, y + 1, 5)] = f2[index_pop_1 + 4];
            f1[scalar_index_pop(x - 1, y + 1, 6)] = f2[index_pop_1 + 5];
            f1[scalar_index_pop(x - 1, y - 1, 7)] = f2[index_pop_1 + 6];
            f1[scalar_index_pop(x + 1, y - 1, 8)] = f2[index_pop_1 + 7];
        }

    // stream left and right boundary nodes
    for (unsigned int y = 1; y < N_Y - 1; y++)  // exclude corners
    {
        int x = 0;
        size_t index_pop_1 = scalar_index_pop(x, y, 1);
        // stream west wall's populations
        f1[scalar_index_pop(x + 1, y    , 1)] = f2[index_pop_1];
        f1[scalar_index_pop(x    , y + 1, 2)] = f2[index_pop_1 + 1];
        f1[scalar_index_pop(x    , y - 1, 4)] = f2[index_pop_1 + 3];
        f1[scalar_index_pop(x + 1, y + 1, 5)] = f2[index_pop_1 + 4];
        f1[scalar_index_pop(x + 1, y - 1, 8)] = f2[index_pop_1 + 7];

        // stream east wall's populations
        x = N_X - 1;
        index_pop_1 = scalar_index_pop(x, y, 1);
        f1[scalar_index_pop(x    , y + 1, 2)] = f2[index_pop_1 + 1];
        f1[scalar_index_pop(x - 1, y    , 3)] = f2[index_pop_1 + 2];
        f1[scalar_index_pop(x    , y - 1, 4)] = f2[index_pop_1 + 3];
        f1[scalar_index_pop(x - 1, y + 1, 6)] = f2[index_pop_1 + 5];
        f1[scalar_index_pop(x - 1, y - 1, 7)] = f2[index_pop_1 + 6];
    }

    // stream bottom and top boundary nodes
    for (unsigned int x = 1; x < N_X - 1; x++)  // exclude corners
    {
        int y = 0;
        size_t index_pop_1 = scalar_index_pop(x, y, 1);
        // stream south wall's populations
        f1[scalar_index_pop(x + 1, y    , 1)] = f2[index_pop_1];
        f1[scalar_index_pop(x    , y + 1, 2)] = f2[index_pop_1 + 1];
        f1[scalar_index_pop(x - 1, y    , 3)] = f2[index_pop_1 + 2];
        f1[scalar_index_pop(x + 1, y + 1, 5)] = f2[index_pop_1 + 4];
        f1[scalar_index_pop(x - 1, y + 1, 6)] = f2[index_pop_1 + 5];

        // stream north wall's populations
        y = N_Y - 1;
        index_pop_1 = scalar_index_pop(x, y, 1);
        f1[scalar_index_pop(x + 1, y    , 1)] = f2[index_pop_1];
        f1[scalar_index_pop(x - 1, y    , 3)] = f2[index_pop_1 + 2];
        f1[scalar_index_pop(x    , y - 1, 4)] = f2[index_pop_1 + 3];
        f1[scalar_index_pop(x - 1, y - 1, 7)] = f2[index_pop_1 + 6];
        f1[scalar_index_pop(x + 1, y - 1, 8)] = f2[index_pop_1 + 7];
    }

    // SW
    int x = 0;
    int y = 0;
    f1[scalar_index_pop(x + 1, y    , 1)] = f2[scalar_index_pop(x, y, 1)];
    f1[scalar_index_pop(x    , y + 1, 2)] = f2[scalar_index_pop(x, y, 2)];
    f1[scalar_index_pop(x + 1, y + 1, 5)] = f2[scalar_index_pop(x, y, 5)];

    // NW
    x = 0;
    y = N_Y - 1;
    f1[scalar_index_pop(x + 1, y    , 1)] = f2[scalar_index_pop(x, y, 1)];
    f1[scalar_index_pop(x    , y - 1, 4)] = f2[scalar_index_pop(x, y, 4)];
    f1[scalar_index_pop(x + 1, y - 1, 8)] = f2[scalar_index_pop(x, y, 8)];

    // NE
    x = N_X - 1;
    y = N_Y - 1;
    f1[scalar_index_pop(x - 1, y, 3)] = f2[scalar_index_pop(x, y, 3)];
    f1[scalar_index_pop(x, y - 1, 4)] = f2[scalar_index_pop(x, y, 4)];
    f1[scalar_index_pop(x - 1, y - 1, 7)] = f2[scalar_index_pop(x, y, 7)];

    // SE
    x = N_X - 1;
    y = 0;
    f1[scalar_index_pop(x, y + 1, 2)] = f2[scalar_index_pop(x, y, 2)];
    f1[scalar_index_pop(x - 1, y, 3)] = f2[scalar_index_pop(x, y, 3)];
    f1[scalar_index_pop(x - 1, y + 1, 6)] = f2[scalar_index_pop(x, y, 6)];
    
}


void streaming_generic(double * f1, double * f2)
{
    // f1[x][y][i] = f2[x - c[i]][y - c[i]][i]
    
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
            for (unsigned int i = 1; i < Q; i++)
            {
                int pos_x = x - c_x[i];
                int pos_y = y - c_y[i];
                if (pos_x < N_X && pos_x >= 0 && pos_y < N_Y && pos_y >= 0)
                    f1[scalar_index_pop(x, y, i)] = f2[scalar_index_pop(pos_x, pos_y, i)];
            }
}


void macr_collision_streaming(double * f1, double * f2, double * f_0, double * rho, double * u_x, double * u_y)
{
    double f_eq[9];
    double T_OMEGA = 1 - OMEGA;
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            size_t index2d = scalar_index2d(x, y);
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            
            // calc for macroscopics
            // rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            // ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) / rho
            // uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) / rho
            rho[index2d] = f_0[index2d] + f1[index_pop_1] + f1[index_pop_1 + 1] + f1[index_pop_1 + 2] + f1[index_pop_1 + 3]
                + f1[index_pop_1 + 4] + f1[index_pop_1 + 5] + f1[index_pop_1 + 6] + f1[index_pop_1 + 7];
            u_x[index2d] = ((f1[index_pop_1] + f1[index_pop_1 + 4] + f1[index_pop_1 + 7]) -
                (f1[index_pop_1 + 2] + f1[index_pop_1 + 5] + f1[index_pop_1 + 6])) / rho[index2d];
            u_y[index2d] = ((f1[index_pop_1 + 1] + f1[index_pop_1 + 4] + f1[index_pop_1 + 5]) -
                (f1[index_pop_1 + 3] + f1[index_pop_1 + 6] + f1[index_pop_1 + 7])) / rho[index2d];
            
            /*
            double rho_v = f_0[index2d] + f1[index_pop_1] + f1[index_pop_1 + 1] + f1[index_pop_1 + 2] + f1[index_pop_1 + 3]
                + f1[index_pop_1 + 4] + f1[index_pop_1 + 5] + f1[index_pop_1 + 6] + f1[index_pop_1 + 7];
            double ux_v = ((f1[index_pop_1] + f1[index_pop_1 + 4] + f1[index_pop_1 + 7]) -
                (f1[index_pop_1 + 2] + f1[index_pop_1 + 5] + f1[index_pop_1 + 6])) / rho[index2d];
            double uy_v = u_y[index2d] = ((f1[index_pop_1 + 1] + f1[index_pop_1 + 4] + f1[index_pop_1 + 5]) -
                (f1[index_pop_1 + 3] + f1[index_pop_1 + 6] + f1[index_pop_1 + 7])) / rho[index2d];
            */
            
            double u_u = u_x[index2d] * u_x[index2d] + u_y[index2d] * u_y[index2d];
            double u_u_cs = u_u * C_S_INV_SQR / 2;
            double rho_w0 = rho[index2d] * W_0;
            double rho_w1 = rho[index2d] * W_1;
            double rho_w2 = rho[index2d] * W_2;
            
            
            /*
            double u_u = ux_v * ux_v + uy_v * uy_v;
            double u_u_cs = u_u * C_S_INV_SQR / 2;
            double rho_w0 = rho_v * W_0;
            double rho_w1 = rho_v * W_1;
            double rho_w2 = rho_v * W_2;
            */

            // calc for f_eq: 
            // f_eq[i] = rho * w[i] (1 + (u-c[i]) / (c_s)^2 + (u-c[i])^2 / (2 * (c_s)^2) + (u-u) / (2 * (c_s)^2))
            // a-b: dot product of a and b; w[i]: velocity weight;  c_s: sound velocity

            f_eq[0] = rho_w0 * (1 - u_u * C_S_INV_SQR / 2.0);

            f_eq[1] = rho_w1 * (1 + u_x[index2d] * C_S_INV_SQR + ((u_x[index2d]) * (u_x[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[2] = rho_w1 * (1 + u_y[index2d] * C_S_INV_SQR + ((u_y[index2d]) * (u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[3] = rho_w1 * (1 - u_x[index2d] * C_S_INV_SQR + ((u_x[index2d]) * (u_x[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[4] = rho_w1 * (1 - u_y[index2d] * C_S_INV_SQR + ((u_y[index2d]) * (u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[5] = rho_w2 * (1 + ( u_x[index2d] + u_y[index2d]) * C_S_INV_SQR + (( u_x[index2d] + u_y[index2d]) * ( u_x[index2d] + u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[6] = rho_w2 * (1 + (-u_x[index2d] + u_y[index2d]) * C_S_INV_SQR + ((-u_x[index2d] + u_y[index2d]) * (-u_x[index2d] + u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[7] = rho_w2 * (1 + (-u_x[index2d] - u_y[index2d]) * C_S_INV_SQR + ((-u_x[index2d] - u_y[index2d]) * (-u_x[index2d] - u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);
            f_eq[8] = rho_w2 * (1 + ( u_x[index2d] - u_y[index2d]) * C_S_INV_SQR + (( u_x[index2d] - u_y[index2d]) * ( u_x[index2d] - u_y[index2d]) * C_S_INV_SQR * C_S_INV_SQR / 2) - u_u_cs);

            // f* = (1 - 1 / TAU) * f + (1 / TAU) * f_eq
            f_0[index2d] = T_OMEGA * f_0[index2d] + OMEGA * f_eq[0];
            f1[index_pop_1] = T_OMEGA * f1[index_pop_1] + OMEGA * f_eq[1];
            f1[index_pop_1 + 1] = T_OMEGA * f1[index_pop_1 + 1] + OMEGA * f_eq[2];
            f1[index_pop_1 + 2] = T_OMEGA * f1[index_pop_1 + 2] + OMEGA * f_eq[3];
            f1[index_pop_1 + 3] = T_OMEGA * f1[index_pop_1 + 3] + OMEGA * f_eq[4];
            f1[index_pop_1 + 4] = T_OMEGA * f1[index_pop_1 + 4] + OMEGA * f_eq[5];
            f1[index_pop_1 + 5] = T_OMEGA * f1[index_pop_1 + 5] + OMEGA * f_eq[6];
            f1[index_pop_1 + 6] = T_OMEGA * f1[index_pop_1 + 6] + OMEGA * f_eq[7];
            f1[index_pop_1 + 7] = T_OMEGA * f1[index_pop_1 + 7] + OMEGA * f_eq[8];

            for (unsigned int i = 1; i < Q; i++)
            {
                int pos_x = (N_X + x + c_x[i]) % N_X;
                int pos_y = (N_Y + y + c_y[i]) % N_Y;
                f2[scalar_index_pop(pos_x, pos_y, i)] = f1[scalar_index_pop(x, y, i)];
            }
        }
}


void boundary_conditions_nebb_pp2d(double* f, double* f0, double* f_post, double* rho, double* u_x, double* u_y)
{
    double u_u;
    unsigned int y;
    
    // special condition for y = 0
    /*
    y = 0;
    u_u = u_x[N_X - 1][y] * u_x[N_X - 1][y] + u_y[N_X - 1][y] * u_y[N_X - 1][y];
    f[0][y][0] = f_eq_i(RHO_IN, u_x[N_X - 1][y], u_u, w[1]) + f_post[N_X - 1][y][0] - f_eq_i(rho[N_X - 1][y], u_x[N_X - 1][y], u_u, w[1]);
    f[0][y][7] = f_eq_i(RHO_IN, u_x[N_X - 1][y] - u_y[N_X - 1][y], u_u, w[8]) + (f_post[N_X - 1][y + 1][7] - f_eq_i(rho[N_X - 1][y], u_x[N_X - 1][y] - u_y[N_X - 1][y], u_u, w[8]));
    u_u = u_x[scalar_index2d(0, y)] * u_x[scalar_index2d(0, y)] + u_y[scalar_index2d(0, y)] * u_y[scalar_index2d(0, y)];
    f[N_X - 1][y][2] = f_eq_i(RHO_OUT, -u_x[scalar_index2d(0, y)], u_u, w[3]) + (f_post[0][y][2] - f_eq_i(rho[scalar_index2d(0, y)], -u_x[scalar_index2d(0, y)], u_u, w[3]));
    f[N_X - 1][y][6] = f_eq_i(RHO_OUT, -u_x[scalar_index2d(0, y)] - u_y[scalar_index2d(0, y)], u_u, w[7]) + (f_post[0][y][6] - f_eq_i(rho[scalar_index2d(0, y)], -u_x[scalar_index2d(0, y)] - u_y[scalar_index2d(0, y)], u_u, w[7]));

    // special condition for y = N_Y - 1
    y = N_Y - 1;
    u_u = u_x[N_X - 1][y] * u_x[N_X - 1][y] + u_y[N_X - 1][y] * u_y[N_X - 1][y];
    f[0][y][0] = f_eq_i(RHO_IN, u_x[N_X - 1][y], u_u, w[1]) + f_post[N_X - 1][y][0] - f_eq_i(rho[N_X - 1][y], u_x[N_X - 1][y], u_u, w[1]);
    f[0][y][4] = f_eq_i(RHO_IN, u_x[N_X - 1][y] + u_y[N_X - 1][y], u_u, w[5]) + (f_post[N_X - 1][y - 1][4] - f_eq_i(rho[N_X - 1][y], u_x[N_X - 1][y] + u_y[N_X - 1][y], u_u, w[5]));
    u_u = u_x[scalar_index2d(0, y)] * u_x[scalar_index2d(0, y)] + u_y[scalar_index2d(0, y)] * u_y[scalar_index2d(0, y)];
    f[N_X - 1][y][2] = f_eq_i(RHO_OUT, -u_x[scalar_index2d(0, y)], u_u, w[3]) + (f_post[0][y][2] - f_eq_i(rho[scalar_index2d(0, y)], -u_x[scalar_index2d(0, y)], u_u, w[3]));
    f[N_X - 1][y][5] = f_eq_i(RHO_OUT, -u_x[scalar_index2d(0, y)] + u_y[scalar_index2d(0, y)], u_u, w[6]) + (f_post[0][y][5] - f_eq_i(rho[scalar_index2d(0, y)], -u_x[scalar_index2d(0, y)] + u_y[scalar_index2d(0, y)], u_u, w[6]));
    */

    // boundary conditions for east and west "walls"
    for (y = 0; y < N_Y; y++)
    {
        // west "wall"
        int x = N_X - 1;
        size_t index2d = scalar_index2d(x, y);
        u_u = u_x[index2d] * u_x[index2d] + u_y[index2d] * u_y[index2d];
        f[scalar_index_pop(0, y, 1)] = f_eq(RHO_IN, u_x[index2d], u_u, w[1]) + 
            (f_post[scalar_index_pop(x, y, 1)] - f_eq(rho[index2d], u_x[index2d], u_u, w[1]));
        if(y > 0)
            f[scalar_index_pop(0, y, 5)] = f_eq(RHO_IN, u_x[index2d] + u_y[index2d], u_u, w[5]) +
            (f_post[scalar_index_pop(x, y - 1, 5)] - f_eq(rho[index2d], u_x[index2d] + u_y[index2d], u_u, w[5]));
        if(y < N_Y - 1)
            f[scalar_index_pop(0, y, 8)] = f_eq(RHO_IN, u_x[index2d] - u_y[index2d], u_u, w[8]) +
            (f_post[scalar_index_pop(x, y + 1, 8)] - f_eq(rho[index2d], u_x[index2d] - u_y[index2d], u_u, w[8]));

        // east "wall"
        index2d = scalar_index2d(0, y);
        u_u = u_x[index2d] * u_x[index2d] + u_y[index2d] * u_y[index2d];
        f[scalar_index_pop(x, y, 3)] = f_eq(RHO_OUT, - u_x[index2d], u_u, w[3]) + 
            (f_post[scalar_index_pop(0, y, 3)] - f_eq(rho[index2d], - u_x[index2d], u_u, w[3]));
        if(y > 0)
            f[scalar_index_pop(x, y, 6)] = f_eq(RHO_OUT, - u_x[index2d] + u_y[index2d], u_u, w[6]) + 
            (f_post[scalar_index_pop(0, y - 1, 6)] - f_eq(rho[index2d], - u_x[index2d] + u_y[index2d], u_u, w[6]));
        if(y < N_Y - 1)
            f[scalar_index_pop(x, y, 7)] = f_eq(RHO_OUT, - u_x[index2d] - u_y[index2d], u_u, w[7]) + 
            (f_post[scalar_index_pop(0, y + 1, 7)] - f_eq(rho[index2d], - u_x[index2d] - u_y[index2d], u_u, w[7]));
    }

    // boundary conditions for south and north walls
    for (unsigned int x = 0; x < N_X; x++)
    {
        // north wall, simplified because wall's u = (0, 0)
        /*
        y = N_Y - 1;
        f[scalar_index_pop(x, y, 4)] = f[scalar_index_pop(x, y, 2)];
        f[scalar_index_pop(x, y, 7)] = f[scalar_index_pop(x, y, 5)] + 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);
        f[scalar_index_pop(x, y, 8)] = f[scalar_index_pop(x, y, 6)] - 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);

        // south wall, simplified because wall's u = (0, 0)
        y = 0;
        f[scalar_index_pop(x, y, 2)] = f[scalar_index_pop(x, y, 4)];
        f[scalar_index_pop(x, y, 5)] = f[scalar_index_pop(x, y, 7)] - 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);
        f[scalar_index_pop(x, y, 6)] = f[scalar_index_pop(x, y, 8)] + 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);
        */

        y = N_Y - 1;
        boundary_conditions_N(&f[scalar_index_pop(x, y, 1)], f0[scalar_index2d(x, y)], 0, 0);
        y = 0;
        boundary_conditions_S(&f[scalar_index_pop(x, y, 1)], f0[scalar_index2d(x, y)], 0, 0);
    }
}


void boundary_conditions_nebb_c2d(double* f, double* f_0)
{
    double rho_w;
    // boundary conditions for east and west walls
    for (unsigned int y = 1; y < N_Y - 1; y++) // exclude corners
    {
        // east wall, simplified because wall's u = (0, 0)
        int x = N_X - 1;
        f[scalar_index_pop(x, y, 3)] = f[scalar_index_pop(x, y, 1)];
        f[scalar_index_pop(x, y, 6)] = f[scalar_index_pop(x, y, 8)] - 0.5 * (f[scalar_index_pop(x, y, 2)] - f[scalar_index_pop(x, y, 4)]);
        f[scalar_index_pop(x, y, 7)] = f[scalar_index_pop(x, y, 5)] + 0.5 * (f[scalar_index_pop(x, y, 2)] - f[scalar_index_pop(x, y, 4)]);

        // west wall, simplified because wall's u = (0, 0)
        x = 0;
        f[scalar_index_pop(x, y, 1)] = f[scalar_index_pop(x, y, 3)];
        f[scalar_index_pop(x, y, 5)] = f[scalar_index_pop(x, y, 7)] - 0.5 * (f[scalar_index_pop(x, y, 2)] - f[scalar_index_pop(x, y, 4)]);
        f[scalar_index_pop(x, y, 8)] = f[scalar_index_pop(x, y, 6)] + 0.5 * (f[scalar_index_pop(x, y, 2)] - f[scalar_index_pop(x, y, 4)]);
    }

    // boundary conditions for south and north walls
    for (unsigned int x = 1; x < N_X - 1; x++) // exclude corners
    {
        // north wall, simplified because wall's u_y = 0
        int y = N_Y - 1;
        rho_w = f_0[scalar_index2d(x, y)] + f[scalar_index_pop(x, y, 1)] + f[scalar_index_pop(x, y, 3)] + 
            2.0 * (f[scalar_index_pop(x, y, 2)] + f[scalar_index_pop(x, y, 5)] + f[scalar_index_pop(x, y, 6)]);

        f[scalar_index_pop(x, y, 4)] = f[scalar_index_pop(x, y, 2)];
        f[scalar_index_pop(x, y, 7)] = f[scalar_index_pop(x, y, 5)] + 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]) - 0.5 * rho_w * U_W_TOP;
        f[scalar_index_pop(x, y, 8)] = f[scalar_index_pop(x, y, 6)] - 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]) + 0.5 * rho_w * U_W_TOP;

        // south wall, simplified because wall's u = (0, 0)
        y = 0;
        f[scalar_index_pop(x, y, 2)] = f[scalar_index_pop(x, y, 4)];
        f[scalar_index_pop(x, y, 5)] = f[scalar_index_pop(x, y, 7)] - 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);
        f[scalar_index_pop(x, y, 6)] = f[scalar_index_pop(x, y, 8)] + 0.5 * (f[scalar_index_pop(x, y, 1)] - f[scalar_index_pop(x, y, 3)]);

    }
    
    // boundary conditions at the corners

    // NE, simplified because corner's u_y = 0
    int x = N_X - 1;
    int y = N_Y - 1;
    rho_w = (f_0[scalar_index2d(x, y)] + 2.0 * f[scalar_index_pop(x, y, 1)] + 2.0 * f[scalar_index_pop(x, y, 2)] 
        + 2.0 * f[scalar_index_pop(x, y, 5)]) / (1.0 + (5.0/6) * U_W_TOP);

    f[scalar_index_pop(x, y, 3)] = f[scalar_index_pop(x, y, 1)] - (2.0 / 3) * rho_w * U_W_TOP;
    f[scalar_index_pop(x, y, 4)] = f[scalar_index_pop(x, y, 2)];
    f[scalar_index_pop(x, y, 7)] = f[scalar_index_pop(x, y, 5)] - (1.0 / 6) * rho_w * U_W_TOP;
    //f[N_X - 1][N_Y - 1][5] = 0;       // do not propagate to fluid
    //f[N_X - 1][N_Y - 1][7] = 0;       // do not propagate to fluid


    // SE, simplified because corner's u = (0, 0)
    // rho_w = f_0[N_X - 1][0] + 2.0 * f[N_X - 1][0][0] + 2.0 * f[N_X - 1][0][3] + 2.0 * f[N_X - 1][0][7];
    x = N_X - 1;
    y = 0;
    f[scalar_index_pop(x, y, 2)] = f[scalar_index_pop(x, y, 4)];
    f[scalar_index_pop(x, y, 3)] = f[scalar_index_pop(x, y, 1)];
    f[scalar_index_pop(x, y, 6)] = f[scalar_index_pop(x, y, 8)];
    //f[N_X - 1][0][4] = 0;         // do not propagate to fluid
    //f[N_X - 1][0][6] = 0;         // do not propagate to fluid


    // SW, simplified because corner's u = (0, 0)
    // rho_w = f_0[0][0] + 2.0 * f[0][0][2] + 2.0 * f[0][0][3] + 2.0 * f[0][0][6];
    
    x = 0;
    y = 0;
    f[scalar_index_pop(x, y, 1)] = f[scalar_index_pop(x, y, 3)];
    f[scalar_index_pop(x, y, 2)] = f[scalar_index_pop(x, y, 4)];
    f[scalar_index_pop(x, y, 5)] = f[scalar_index_pop(x, y, 7)];
    //f[0][0][5] = 0;                   // do not propagate to fluid
    //f[0][0][7] = 0;                   // do not propagate to fluid


    // NW, simplified because corner's u_y = 0
    x = 0;
    y = N_Y - 1;
    rho_w = (f_0[scalar_index2d(x, y)] + 2.0 * f[scalar_index_pop(x, y, 2)] + 2.0 * f[scalar_index_pop(x, y, 3)]
        + 2.0 * f[scalar_index_pop(x, y, 6)]) / (1.0 - (5.0 / 6) * U_W_TOP);
    f[scalar_index_pop(x, y, 1)] = f[scalar_index_pop(x, y, 3)] + (2.0 / 3) * rho_w * U_W_TOP;
    f[scalar_index_pop(x, y, 4)] = f[scalar_index_pop(x, y, 2)];
    f[scalar_index_pop(x, y, 8)] = f[scalar_index_pop(x, y, 6)] + (1.0 / 6) * rho_w * U_W_TOP;
    //f[0][N_Y - 1][4] = 0;         // do not propagate to fluid
    //f[0][N_Y - 1][6] = 0;         // do not propagate to fluid
}


void update_rho_u(double* f, double* f_0, double* rho, double* u_x, double* u_y)
{
    for (unsigned int x = 0; x < N; x++)
        for (unsigned int y = 0; y < N; y++)
        {
            size_t index2d = scalar_index2d(x, y);
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            // rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            rho[index2d] = f_0[index2d] + f[index_pop_1] + f[index_pop_1 + 1] + f[index_pop_1 + 2] + f[index_pop_1 + 3]
                + f[index_pop_1 + 4] + f[index_pop_1 + 5] + f[index_pop_1 + 6] + f[index_pop_1 + 7];
            
            // ux = ((f1 + f5 + f8) - (f3 + f6 + f7)) / (f0 + f1 + ... + f8)
            u_x[index2d] = ((f[index_pop_1] + f[index_pop_1 + 4] + f[index_pop_1 + 7]) -
                (f[index_pop_1 + 2] + f[index_pop_1 + 5] + f[index_pop_1 + 6])) / rho[index2d];
            
            // uy = ((f2 + f5 + f6) - (f4 + f7 + f8)) / (f0 + f1 + ... + f8)
            u_y[index2d] = ((f[index_pop_1 + 1] + f[index_pop_1 + 4] + f[index_pop_1 + 5]) -
                (f[index_pop_1 + 3] + f[index_pop_1 + 6] + f[index_pop_1 + 7])) / rho[index2d];
        }
}


void update_rho_u_generic(double * f, double * f_0, double * rho, double * u_x, double * u_y)
{
    for (unsigned int x = 0; x < N; x++)
        for (unsigned int y = 0; y < N; y++)
        {
            double sum_f = 0, sum_fx = 0, sum_fy = 0;
            size_t index2d = scalar_index2d(x, y);
            size_t index_pop_1 = scalar_index_pop(x, y, 1);
            sum_f += f_0[index2d];
            for (unsigned int i = 1; i < Q; i++)
            {
                sum_f += f[index_pop_1 + i - 1];
                sum_fx += f[index_pop_1 + i - 1] * c_x[i];
                sum_fy += f[index_pop_1 + i - 1] * c_y[i];
            }
            rho[index2d] = sum_f;
            u_x[index2d] = sum_fx / sum_f;
            u_y[index2d] = sum_fy / sum_f;
        }
}


const double residual(double* u_x, double* u_y, double* u_x_res, double* u_y_res)
{
    double den = 0.0, num = 0.0;

    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            const double diff_ux = u_x[scalar_index2d(x, y)] - u_x_res[scalar_index2d(x, y)];
            const double diff_uy = u_y[scalar_index2d(x, y)] - u_y_res[scalar_index2d(x, y)];

            num += std::sqrt(diff_ux * diff_ux + diff_uy * diff_uy);
            den += std::sqrt(u_x[scalar_index2d(x, y)] * u_x[scalar_index2d(x, y)] + u_y[scalar_index2d(x, y)] * u_y[scalar_index2d(x, y)]);
        }
    return (num / den);
}


const double u_med(double* u_x)
{
    double u_sum = 0.0;
    for (int y = 0; y < N_Y; y++)
    {
        u_sum += u_x[scalar_index2d(N_X / 2, y)];
    }
    return (u_sum / (N_Y - 1));
}


void equalize_vel(double* u_x, double* u_y, double* u_x_0, double* u_y_0)
{
    for (unsigned int x = 0; x < N_X; x++)
        for (unsigned int y = 0; y < N_Y; y++)
        {
            u_x_0[scalar_index2d(x, y)] = u_x[scalar_index2d(x, y)];
            u_y_0[scalar_index2d(x, y)] = u_y[scalar_index2d(x, y)];
        }
}


void save(const std::string id, const unsigned int n_steps, double* rho, double* u_x, double* u_y)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
    if (n_steps != 0)
        for (n_zeros = 0; n_steps * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 4;

    // generate file in format "version_u0000", with the number being n_steps
    std::string str_ux = PATH_DATA + id + "_ux";
    std::string str_uy = PATH_DATA + id + "_uy";
    std::string str_rho = PATH_DATA + id + "_rho";
    for (unsigned int i = 0; i < n_zeros; i++)
    {
        str_ux += "0";
        str_uy += "0";
        str_rho += "0";
    }
    str_ux += std::to_string(n_steps) + EXT;
    str_uy += std::to_string(n_steps) + EXT;
    str_rho += std::to_string(n_steps) + EXT;

    std::fstream outFile_ux(str_ux.c_str(), std::fstream::out);
    std::fstream outFile_uy(str_uy.c_str(), std::fstream::out);
    std::fstream outFile_rho(str_rho.c_str(), std::fstream::out);

    // save u and rho csv data, y iteration is backwards to save top down
    for (int y = N_Y - 1; y >= 0; y--)
    {
        for (unsigned int x = 0; x < N_X; x++)
        {
            // fix precision to 6 houses and scientific notation
            // write rho
            outFile_rho << std::fixed << std::scientific << std::setprecision(6);
            outFile_rho << rho[scalar_index2d(x, y)] << SEP;
            // write u_x
            outFile_ux << std::fixed << std::scientific << std::setprecision(6);
            outFile_ux << u_x[scalar_index2d(x, y)] << SEP;
            // write u_y
            outFile_uy << std::fixed << std::scientific << std::setprecision(6);
            outFile_uy << u_y[scalar_index2d(x, y)] << SEP;
        }
        outFile_rho << std::endl;
        outFile_ux << std::endl;
        outFile_uy << std::endl;
    }

    outFile_ux.close();
    outFile_uy.close();
    outFile_rho.close();
}


void save_inf(const std::string id, const double mlups, const double res, const int n_steps)
{
    std::string str_inf = PATH_DATA + id + "_inf.txt"; // generate file name (with path)

    std::fstream outFile_inf(str_inf.c_str(), std::fstream::out);

    // writes n_x, n_y, tau and the top wall's velocity
    outFile_inf << "Program's ID = " << ID_PROG << std::endl;
    outFile_inf << "n_x = " << N_X << std::endl;
    outFile_inf << "n_y = " << N_Y << std::endl;
    outFile_inf << "Reynolds = " << REYNOLDS << std::endl;
    outFile_inf << "Tau = " << TAU << std::endl;
    outFile_inf << "u_max = " << U_W_TOP << std::endl;
    outFile_inf << "residual = " << res << std::endl;
    outFile_inf << "MLups = " << mlups << std::endl;
    outFile_inf << "n_steps = " << n_steps << std::endl;

    outFile_inf.close();
}


void save_ux_uy(const std::string id, double* u_x, double* u_y)
{
    std::string str_ux = PATH_DATA + id + "_ux_c" + EXT;    // generate u_x file name (with path)
    std::string str_uy = PATH_DATA + id + "_uy_c" + EXT;    // generate u_y file name (with path)
    
    std::fstream outFile_ux(str_ux.c_str(), std::fstream::out); 
    std::fstream outFile_uy(str_uy.c_str(), std::fstream::out);

    int x, y;

    for (x = 0; x < N_X; x++)
    {
        // if the number of nodes is even, the value saved is de average of the ones indexes [y][N_Y/2] and [x][N_Y/2-1]
        double u_y_i;
        if (N_Y % 2)
            u_y_i = u_y[scalar_index2d(x, N_Y / 2)];
        else
            u_y_i = (u_y[scalar_index2d(x, N_Y / 2)] + u_y[scalar_index2d(x, N_Y / 2 - 1)]) / 2;
        
        // fix precision to 6 houses
        outFile_uy << std::fixed << std::setprecision(6);
        outFile_uy << ((double)x / (N_X - 1)) << SEP;   
        // fix scientific notation
        outFile_uy << std::scientific;
        outFile_uy << u_y_i/U_W_TOP << std::endl;       // writes normalized velocity
    }

    for (y = 0; y < N_Y; y++)
    {
        // if the number of nodes is even, the value saved is de average of the ones indexes [N_X/2][y] and [N_X/2-1][y]
        double u_x_i;
        if (N_X % 2)
            u_x_i = u_x[scalar_index2d(N_X / 2, y)];
        else
            u_x_i = (u_x[scalar_index2d(N_X / 2, y)] + u_x[scalar_index2d(N_X / 2 - 1, y)]) / 2;

        outFile_ux << std::fixed << std::setprecision(6);
        outFile_ux << ((double)y / (N_Y - 1)) << SEP;
        outFile_ux << std::scientific;
        outFile_ux << u_x_i/U_W_TOP << std::endl;   // writes normalized velocity

    }

    outFile_ux.close();
    outFile_uy.close();
}


void save_ux(const std::string id, double* u_x)
{
    std::string str_ux = PATH_DATA + id + "_ux_c" + EXT;    // generate u_x file name (with path)

    std::fstream outFile_ux(str_ux.c_str(), std::fstream::out);

    for (int y = 0; y < N_Y; y++)
    {
        // if the number of nodes is even, the value saved is de average of the ones indexes [N_X/2][y] and [N_X/2-1][y]
        double u_x_i;
        if (N_X % 2)
            u_x_i = u_x[scalar_index2d(N_X / 2, y)];
        else
            u_x_i = (u_x[scalar_index2d(N_X / 2, y)] + u_x[scalar_index2d(N_X / 2 - 1, y)]) / 2;

        outFile_ux << std::fixed << std::setprecision(6);
        outFile_ux << ((double)y / (N_Y - 1)) << SEP;
        outFile_ux << std::scientific;
        outFile_ux << u_x_i / U_W_TOP << std::endl; // writes normalized velocity

    }

    outFile_ux.close();
}
