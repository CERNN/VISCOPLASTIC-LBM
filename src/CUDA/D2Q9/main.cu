
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
#include ".\..\common\seconds.h"
#include ".\..\common\lbm_save.h"
#include "lbm_d2q9.cuh"

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
    printf("           Reynolds: %.2f\n", REYNOLDS);
    printf("                Tau: %.2f\n", TAU);
    printf("              U max: %.2f\n", U_MAX);
    printf("\n");

    dfloat bytesPerMiB = 1024.0*1024.0;
    dfloat bytesPerGiB = 1024.0*1024.0*1024.0;
    size_t total_mem_bytes = 2*mem_size_pop + 3*mem_size_scalar; // DESATUALIZADO (ADICONAR BC)

    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n", deviceProp.name);
    printf("    multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("      global memory: %.1f MiB\n", deviceProp.totalGlobalMem / bytesPerMiB);
    printf("        free memory: %.1f MiB\n", gpu_free_mem / bytesPerMiB);
    printf("\n");

    int i;
    dfloat *f1_gpu, *f2_gpu;                // gpu populations 0-8
    dfloat *rho_gpu, *ux_gpu, *uy_gpu;      // gpu density and velocity
    dfloat *rho = nullptr;                  // density
    dfloat *ux = nullptr, *uy = nullptr;    // velocities
    NodeTypeMap* nt_map_gpu;                // gpu node type map (for boundary conditions)
    dfloat resid = 1.0f, t0, t1;                // residual (L2 error) and time variables
    dfloat* u_x_res = nullptr, *u_y_res = nullptr;  // residual velocities

	
    // allocate memory from device and host
    checkCudaErrors(cudaMalloc((void**)&f1_gpu, mem_size_pop));
    checkCudaErrors(cudaMalloc((void**)&f2_gpu, mem_size_pop));
    checkCudaErrors(cudaMalloc((void**)&rho_gpu, mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&ux_gpu, mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&uy_gpu, mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&nt_map_gpu, mem_size_bc_map));
    checkCudaErrors(cudaMallocHost((void**)&rho, mem_size_scalar));
    checkCudaErrors(cudaMallocHost((void**)&ux, mem_size_scalar));
    checkCudaErrors(cudaMallocHost((void**)&uy, mem_size_scalar));
    // residual velocities don't need to be pinned, since there's no copy to device
    u_x_res = new dfloat[N_X * N_Y];
    u_y_res = new dfloat[N_X * N_Y];
    for(int x = 0; x < N_X; x++) // initialize res velocities
        for (int y = 0; y < N_Y; y++)
        {
            u_x_res[index_scalar(x, y)] = 0;
            u_y_res[index_scalar(x, y)] = 0;
        }

    // streams for copying device variables
    cudaStream_t stream[3];
    for (i = 0; i < 3; i++)
        checkCudaErrors(cudaStreamCreate(&stream[i]));

    // create event objects
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    // build node type map
    build_boundary_conditions(nt_map_gpu, BC_USED);

    t0 = seconds();
    checkCudaErrors(cudaEventRecord(start, 0));

    initialisation(f1_gpu, f2_gpu, rho_gpu, ux_gpu, uy_gpu);

    for (i = 1; i <= N_STEPS && (resid >= RESID_MAX || !RESID); i++)
    {
        bool save = N_SAVE && !(i % N_SAVE);
        bool msg = N_MSG && !(i % N_MSG);
        bool res = RESID && !(i % N_RESID);

        bc_macr_collision_streaming(f1_gpu, f2_gpu, rho_gpu, ux_gpu, uy_gpu, nt_map_gpu, save || msg || res);
        dfloat* tmp = f1_gpu;
        f1_gpu = f2_gpu;
        f2_gpu = tmp;

        if (save || res || msg)
        {
            checkCudaErrors(cudaDeviceSynchronize()); // prevent from altering macroscopics before the copy ends
            checkCudaErrors(cudaMemcpyAsync(rho, rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost, stream[0]));
            checkCudaErrors(cudaMemcpyAsync(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost, stream[1]));
            checkCudaErrors(cudaMemcpyAsync(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost, stream[2]));
            checkCudaErrors(cudaDeviceSynchronize()); // prevent from altering macroscopics before the copy ends
        }

        if (save)
        {
            save_variable_bin(ID_SIM, "rho", i, rho, mem_size_scalar);
            save_variable_bin(ID_SIM, "ux", i, ux, mem_size_scalar);
            save_variable_bin(ID_SIM, "uy", i, uy, mem_size_scalar);
        }

        if (res)
        {
            resid = residual(ux, uy, u_x_res, u_y_res);
            equalize_vel(ux, uy, u_x_res, u_y_res);
        }

        if (msg)
        {
            std::cout << std::fixed << std::scientific;
            std::cout << "Iteration " << i << std::endl;
            std::cout << "ux_c = " << ux[index_scalar(N_X/2, N_Y/2)]/U_MAX;
            std::cout << " - uy_c = " << uy[index_scalar(N_X/2, N_Y/2)]/U_MAX;
            std::cout << " - rho_c = " << rho[index_scalar(N_X/2, N_Y/2)];
            std::cout << " - res = " << resid << std::endl;
            std::cout << std::endl;
        }
    }
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    t1 = seconds();
    dfloat time_elapsed = t1 - t0;
    dfloat gpu_time_elapsed = milliseconds * 0.001;

    // saves last data
    if(i > N_STEPS || i%10) 
        i -= 1;


    checkCudaErrors(cudaMemcpy(rho, rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    save_variable_bin(ID_SIM, "rho", i, rho, mem_size_scalar);
    save_variable_bin(ID_SIM, "ux", i, ux, mem_size_scalar);
    save_variable_bin(ID_SIM, "uy", i, uy, mem_size_scalar);

    // calculates million lattice updates per second
    dfloat mlups = (N_X*N_Y / 1e6) * i / (time_elapsed);
    
    // INVALID
    // calculates bandwidht
    dfloat bandwidth;
    size_t dfloats_read = Q;            // per node every time step
    size_t dfloats_written = Q;
    size_t dfloats_saved = 3;           // per node every NSAVE time steps
    size_t nodes_updated = i * size_t(N_X*N_Y);
    size_t nodes_saved;
    if (N_SAVE)
        nodes_saved = (i/N_SAVE) * size_t(N_X*N_Y);
    else
        nodes_saved = 0;
    bandwidth = (nodes_updated*(dfloats_read+dfloats_written) + nodes_saved*(dfloats_saved))*sizeof(dfloat) / (time_elapsed*bytesPerGiB);

    // saves simulation info
    save_sim_inf(ID_SIM, mlups, bandwidth, resid, i, deviceProp);

    // last step info
    std::cout << "Iteration " << i << std::endl;
    std::cout << "ux_c = " << ux[index_scalar(N_X/2, N_Y/2)] / U_MAX;
    std::cout << " - uy_c = " << uy[index_scalar(N_X/2, N_Y/2)] / U_MAX;
    std::cout << " - rho_c = " << rho[index_scalar(N_X/2, N_Y/2)];
    std::cout << " - res = " << resid << std::endl;

    // simulation info again
    printf("SIMULATION INFORMATION\n");
    printf("                 NX: %u\n", N_X);
    printf("                 NY: %u\n", N_Y);
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
    for (i = 0; i < 3; i++)
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    // -------------------------

    // ---- deallocate memory ----
    checkCudaErrors(cudaFree(f1_gpu));
    checkCudaErrors(cudaFree(f2_gpu));
    checkCudaErrors(cudaFree(rho_gpu));
    checkCudaErrors(cudaFree(ux_gpu));
    checkCudaErrors(cudaFree(uy_gpu));
    checkCudaErrors(cudaFree(nt_map_gpu));
    checkCudaErrors(cudaFreeHost(rho));
    checkCudaErrors(cudaFreeHost(ux));
    checkCudaErrors(cudaFreeHost(uy));
    delete(u_x_res);
    delete(u_y_res);
    // ----------------------------

    // release resources associated with the GPU device
    cudaDeviceReset();

    //getchar();
    
    return 1;
}