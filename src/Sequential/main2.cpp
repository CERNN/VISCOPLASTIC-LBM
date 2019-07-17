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

// main.cpp : Este arquivo cont�m a fun��o 'main'

#include "pch.h"

#include "lbm2.h"
#include "seconds.h"

#define PRINT 1

/// defines for test options
constexpr int N_STEPS = 1e6;
#define N_SAVE 10000000
#define N_MSG 1000
#define ID_SIM "CPU_"

int main()
{
    int i;
    double mlups = 0.0, t1 = 0.0, t0 = 0.0;
    double res = 1.0;                               // residual
    double* f = nullptr;                            // populations 1-8
    double* f0 = nullptr;                           // populations 0
    double* f_post = nullptr;                       // populations 1-8 after collision
    double* rho = nullptr;                          // densities
    double* u_x = nullptr, *u_y = nullptr;          // velocities
    double* u_x_res = nullptr, *u_y_res = nullptr;  // residual velocities


    // ---- MEMORY ALLOCATION ----
    f = new double[N_X * N_Y * (Q - 1)];
    f0 = new double[N_X * N_Y];
    f_post = new double[N_X * N_Y * (Q - 1)];
    rho = new double[N_X * N_Y];
    u_x = new double[N_X * N_Y];
    u_y = new double[N_X * N_Y];
    u_x_res = new double[N_X * N_Y];
    u_y_res = new double[N_X * N_Y];

    // ----------------------------

    std::cout << "n_x = " << N_X << std::endl;
    std::cout << "n_y = " << N_Y << std::endl;
    std::cout << "Reynolds = " << REYNOLDS << std::endl;
    std::cout << "Tau = " << TAU << std::endl;
    std::cout << "u_top_wall = " << U_W_TOP << std::endl << std::endl;

    t0 = seconds();

    initialisation(f, f_post, f0, rho, u_x, u_y);

    for (i = 0; res > RESID_MAX && i < N_STEPS; i++)
    {
        //collision(f, f_post, f0, rho, u_x, u_y);
        //streaming(f, f_post);

        macr_collision_streaming(f, f_post, f0, rho, u_x, u_y);
        double* tmp = f;
        f = f_post;
        f_post = tmp;

        boundary_conditions_nebb_pp2d(f, f0, f_post, rho, u_x, u_y);
        //boundary_conditions_nebb_c2d(f, f0);
        
        //update_rho_u(f, f0, rho, u_x, u_y);
        
        
        if (!(i % 1000))
        {
            if(i != 0)
                res = residual(u_x, u_y, u_x_res, u_y_res);
            equalize_vel(u_x, u_y, u_x_res, u_y_res);
        }
        

        if (!((i-1) % N_SAVE) && (i-1))
        {
            save(ID_SIM, i, rho, u_x, u_y);
        }

        if (!(i % N_MSG))
        {
            std::cout << std::fixed << std::scientific << std::setprecision(6);
            std::cout << "Iteration " << i << std::endl; 
            std::cout << "ux_c = " << u_x[scalar_index2d(N_X / 2, N_Y / 2)] / U_W_TOP;
            std::cout << " - uy_c = " << u_y[scalar_index2d(N_X/2, N_Y/2)] / U_W_TOP;
            std::cout << " - rho_c = " << rho[scalar_index2d(N_X / 2, N_Y / 2)];
            std::cout << " - res = " << res << std::endl;
            std::cout << std::endl;
        }
    }
    //update_rho_u(f, f0, rho, u_x, u_y);

    t1 = seconds();
    double time_elapsed = t1 - t0;

    // calculates million lattice updates per second
    mlups = (N_X * N_Y / 1e6) * i / (time_elapsed);
    
    // saves simulation info
    save_inf(ID_SIM, mlups, res, i);

    // saves last data
    // save(ID_SIM, ((i % 10) ? i - 1 : i), rho, u_x, u_y);

    // saves ux x=0.5 and uy y=0.5
    // save_ux_uy(ID_SIM, u_x, u_y);

    // saves ux x=0.5
    save_ux(ID_SIM, u_x);

    // ---- deallocate memory ----
    delete(f);
    delete(f0);
    delete(f_post);
    delete(rho);
    delete(u_x);
    delete(u_y);
    delete(u_x_res);
    delete(u_y_res);
    // ----------------------------
    
    return 1;
}