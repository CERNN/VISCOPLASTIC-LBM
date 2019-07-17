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


#include <iostream>
#include <new>
#include <cuda.h>
#include <iomanip>
#include "./../common/seconds.h"
#include "./../common/lbm_save.h"
#include "lbm_d3q19.cuh"


// TODO: 
int main()
{
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));

    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));

    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // Set cache to 48kB and Shared Memory to 12kB

    printf("SIMULATION INFORMATION\n");
    printf("                 NX: %u\n", N_X);
    printf("                 NY: %u\n", N_Y);
    printf("                 NZ: %u\n", N_Z);
    printf("           Reynolds: %.2f\n", REYNOLDS);
    printf("                Tau: %.2f\n", TAU);
    printf("              U max: %.2f\n", U_MAX);
    printf("             rho in: %.4f\n", RHO_IN);
    printf("            rho out: %.4f\n", RHO_OUT);
    printf("\n");

    dfloat bytesPerMiB = 1024.0*1024.0;
    dfloat bytesPerGiB = 1024.0*1024.0*1024.0;
    size_t total_mem_bytes = 2*mem_size_pop + 4*mem_size_scalar;

    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n", deviceProp.name);
    printf("    multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("      global memory: %.1f MiB\n", deviceProp.totalGlobalMem / bytesPerMiB);
    printf("        free memory: %.1f MiB\n", gpu_free_mem / bytesPerMiB);
    printf("\n");

    int i;
    dfloat *f1 = nullptr, *f2 = nullptr;                            // gpu populations                                                 // variable to save populations
    dfloat *rho = nullptr;                                          // density
    dfloat *ux = nullptr, *uy = nullptr, *uz = nullptr;             // velocities
    NodeTypeMap* nt_map= nullptr;                                   // gpu node type map (for boundary conditions)
    dfloat resid = 1.0f;
    double t0 = 0, t1 = 0;                            // residual (L2 error) and time variables
    dfloat* ux_res = nullptr, *uy_res = nullptr, *uz_res = nullptr; // residual velocities

    // allocate memory from device and host
    checkCudaErrors(cudaMallocManaged((void**)&f1, mem_size_pop));
    checkCudaErrors(cudaMallocManaged((void**)&f2, mem_size_pop));
    checkCudaErrors(cudaMallocManaged((void**)&nt_map, mem_size_bc_map));
    checkCudaErrors(cudaMallocManaged((void**)&rho, mem_size_scalar));
    checkCudaErrors(cudaMallocManaged((void**)&ux, mem_size_scalar));
    checkCudaErrors(cudaMallocManaged((void**)&uy, mem_size_scalar));
    checkCudaErrors(cudaMallocManaged((void**)&uz, mem_size_scalar));
    // residual velocities don't need to be pinned, since there's no copy to device

    
    ux_res = new dfloat[N_X * N_Y * N_Z];
    uy_res = new dfloat[N_X * N_Y * N_Z];
    uz_res = new dfloat[N_X * N_Y * N_Z];
    for (unsigned int z = 0; z < N_Z; z++)
        for (unsigned int y = 0; y < N_Y; y++)
            for (unsigned int x = 0; x < N_X; x++)
            {
                ux_res[index_scalar_d3(x, y, z)] = 0;
                uy_res[index_scalar_d3(x, y, z)] = 0;
                uz_res[index_scalar_d3(x, y, z)] = 0;
            }

    // streams variables
    cudaStream_t stream_lbm;
    checkCudaErrors(cudaStreamCreate(&stream_lbm));

    // create event objects
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    // build node type map
    build_boundary_conditions(nt_map, BC_USED);

    if (LOAD_MACR)
    {
        std::string str_rho = get_var_filename(ID_SIM, "rho", LOAD_MACR);
        std::string str_ux = get_var_filename(ID_SIM, "ux", LOAD_MACR);
        std::string str_uy = get_var_filename(ID_SIM, "uy", LOAD_MACR);
        std::string str_uz = get_var_filename(ID_SIM, "uz", LOAD_MACR);
        
        FILE* file_rho = fopen(str_rho.c_str(), "r");
        FILE* file_ux = fopen(str_ux.c_str(), "r");
        FILE* file_uy = fopen(str_uy.c_str(), "r");
        FILE* file_uz = fopen(str_uz.c_str(), "r");
        macr_initialisation(f1, f2, rho, file_rho, ux, file_ux, 
            uy, file_uy, uz, file_uz);
        fclose(file_rho);
        fclose(file_ux);
        fclose(file_uy);
        fclose(file_uz);
    }
    else if (LOAD_POP)
    {
        std::string str_pop = get_var_filename(ID_SIM, "pop", LOAD_POP);
        FILE* file_pop = fopen(str_pop.c_str(), "r");
        pop_initialisation(f1, file_pop, f2, rho, ux, uy, uz);
        fclose(file_pop);
    }
    else
        initialisation(f1, f2, rho, ux, uy, uz, false, true);
    checkCudaErrors(cudaDeviceSynchronize());

    t0 = seconds();
    checkCudaErrors(cudaEventRecord(start, 0));

    for (i = INI_STEP+1; i <= N_STEPS && (resid >= RESID_MAX || !RESID); i++)
    {
        bool save = false;
        if(N_SAVE != 0)
            save = !(i % N_SAVE);
        
        bool msg = false;
        if(N_MSG != 0)
            msg = !(i % N_MSG);
        
        bool res = false;
        if(N_RESID != 0 && RESID != 0)
            res = !(i % N_RESID);

        bc_macr_collision_streaming(f1, f2, rho, ux,
            uy, uz, nt_map, save || msg || res || (i==N_STEPS), i-INI_STEP, &stream_lbm);

        dfloat* tmp = f1;
        f1 = f2;
        f2 = tmp;

        if (save || res)
        {
            printf("sincronizando na iteracao %d...\n", i);
            checkCudaErrors(cudaStreamSynchronize(stream_lbm));
        }

        if (save)
        {
            printf("salvando...\n");
            save_variable_bin(ID_SIM, "rho", i, rho, mem_size_scalar);
            save_variable_bin(ID_SIM, "ux", i, ux, mem_size_scalar);
            save_variable_bin(ID_SIM, "uy", i, uy, mem_size_scalar);
            save_variable_bin(ID_SIM, "uz", i, uz, mem_size_scalar);
            printf("salvou!\n");
        }

        if (res)
        {
            printf("residuo...\n");
            resid = residual(ux, uy, uz, ux_res, uy_res, uz_res);
            equalize_vel(ux, uy, uz, ux_res, uy_res, uz_res);
            printf("residuo!\n");
        }

        if (msg || res)
        {
            
            std::cout << std::fixed << std::scientific;
            std::cout << "Iteration " << i << std::endl;
            if (save)
            {
                std::cout << " - ux_s = " << ux[index_scalar_d3(N_X / 2, 0, N_Z / 2)] / U_MAX;
                std::cout << " - ux_n = " << ux[index_scalar_d3(N_X / 2, N_Y - 1, N_Z / 2)] / U_MAX << std::endl;
                std::cout << " - uy_s = " << uy[index_scalar_d3(N_X / 2, 0, N_Z/2)] / U_MAX;
                std::cout << " - uy_n = " << uy[index_scalar_d3(N_X / 2, N_Y-1, N_Z/2)] / U_MAX << std::endl;
                std::cout << " - uz_s = " << uz[index_scalar_d3(N_X / 2, 0, N_Z / 2)] / U_MAX;
                std::cout << " - uz_n = " << uz[index_scalar_d3(N_X / 2, N_Y - 1, N_Z / 2)] / U_MAX << std::endl;
                std::cout << " - rho_b = " << rho[index_scalar_d3(N_X / 2, N_Y / 2,0)]; 
                std::cout << " - rho_f = " << rho[index_scalar_d3(N_X / 2, N_Y / 2, N_Z-1)] << " - ";
            }
            if(res)
                std::cout << " res = " << resid << std::endl;
            std::cout << std::endl;
            
        }
    }
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    t1 = seconds();
    double time_elapsed = t1 - t0;
    double gpu_time_elapsed = milliseconds * 0.001;

    // saves last data
    if(i > N_STEPS || i%10) 
        i -= 1;

    checkCudaErrors(cudaStreamSynchronize(stream_lbm));
    if (SAVE_POP) {
        save_variable_bin(ID_SIM, "pop", i, f1, mem_size_pop);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    save_variable_bin(ID_SIM, "rho", i, rho, mem_size_scalar);
    save_variable_bin(ID_SIM, "ux", i, ux, mem_size_scalar);
    save_variable_bin(ID_SIM, "uy", i, uy, mem_size_scalar);
    save_variable_bin(ID_SIM, "uz", i, uz, mem_size_scalar);


    // calculates million lattice updates per second
    double mlups = (N_X*N_Y*N_Z / 1e6) * i / (gpu_time_elapsed);
    
    // INVALID
    // calculates bandwidht
    double bandwidth;
    size_t dfloats_read = Q;            // per node every time step
    size_t dfloats_written = Q;
    size_t dfloats_saved = 4;           // per node every NSAVE time steps
    size_t nodes_updated = (i-INI_STEP) * size_t(N_X*N_Y*N_Z);
    size_t nodes_saved;
    if (N_SAVE)
        nodes_saved = (i/N_SAVE) * size_t(N_X*N_Y*N_Z);
    else
        nodes_saved = 0;
    bandwidth = (nodes_updated*(dfloats_read+dfloats_written) + nodes_saved*(dfloats_saved))*sizeof(dfloat) / (gpu_time_elapsed*bytesPerGiB);

    // saves simulation info
    save_sim_inf_d3(ID_SIM, mlups, bandwidth, resid, i, deviceProp);

    // last step info
    std::cout << "Iteration " << i << std::endl;
    std::cout <<    "ux_c = " << ux[index_scalar_d3(N_X/2, N_Y/2, N_Z/2)] / U_MAX;
    std::cout << " - uy_c = " << uy[index_scalar_d3(N_X/2, N_Y/2, N_Z/2)] / U_MAX;
    std::cout << " - uz_c = " << uz[index_scalar_d3(N_X/2, N_Y/2, N_Z/2)] / U_MAX;
    std::cout << " - rho_c = " << rho[index_scalar_d3(N_X/2, N_Y/2, N_Z/2)];
    std::cout << " - res = " << resid << std::endl;

    // simulation info again
    printf("SIMULATION INFORMATION\n");
    printf("                 NX: %u\n", N_X);
    printf("                 NY: %u\n", N_Y);
    printf("                 NZ: %u\n", N_Y);
    printf("           Reynolds: %.2f\n", REYNOLDS);
    printf("                Tau: %.2f\n", TAU);
    printf("              U max: %.2f\n", U_MAX);
    printf("\n");
    // performance information
    printf("PERFORMANCE INFO:\n");
    printf("               timesteps: %u\n", i);
    printf("           clock runtime: %.3f (s)\n", time_elapsed);
    printf("             gpu runtime: %.3f (s)\n", gpu_time_elapsed);
    printf("                   MLUPS: %.2f (Mlups)\n", mlups);
    printf("               bandwidth: %.1f (GiB/s)\n", bandwidth);

    // ---- destroy streams ----
    checkCudaErrors(cudaStreamDestroy(stream_lbm));
    // -------------------------

    // ---- deallocate memory ----
    checkCudaErrors(cudaFree(f2));
    checkCudaErrors(cudaFree(f1));
    checkCudaErrors(cudaFree(nt_map));
    checkCudaErrors(cudaFree(rho));
    checkCudaErrors(cudaFree(ux));
    checkCudaErrors(cudaFree(uy));
    checkCudaErrors(cudaFree(uz));
    delete(ux_res);
    delete(uy_res);
    delete(uz_res);
    // ----------------------------

    // release resources associated with the GPU device
    cudaDeviceReset();

    //getchar();
    
    return 1;
}
