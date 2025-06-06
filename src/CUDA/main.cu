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

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "treatData.h"
#include "lbmReport.h"
#include "lbm.h"
#include "lbmInitialization.h"
#include "simCheckpoint.h"
#include "boundaryConditionsBuilder.h"
#include "structs/boundaryConditionsInfo.h"

#include "IBM/ibm.h"
#include "IBM/ibmParticlesCreation.h"
#include "IBM/ibmTreatData.h"

#include "AuxFunctions/auxFunction.cuh"

#ifdef LES_MODEL
#include "LES/les.h"
#endif



int main()
{
    // Variables declaration
    Populations* pop;
    Macroscopics* macr;
    Macroscopics macrCPUCurrent;
    Macroscopics macrCPUOld;
    MacrProc processData;
    BoundaryConditionsInfo* bcInfos;
    SimInfo info;

    float** randomNumbers = nullptr; // useful for turbulence
    int step = INI_STEP;
    dim3* gridsBC;

    ParticlesSoA particlesSoA;
    Particle *particles;
    particles = (Particle*) malloc(sizeof(Particle)*NUM_PARTICLES);
    ParticleEulerNodesUpdate pEulerNodes;

    IBMProc ibmProcessData;
    IBMMacrsAux ibmMacrsAux;
    #ifdef IBM
    allocateIBMProc(&ibmProcessData);
    #endif

    #ifdef DENSITY_CORRECTION
    dfloat h_mean_rho[N_GPUS];
    dfloat* d_mean_rho;
    #endif

    // Setup saving folder
    folderSetup();

    // Initializes process data
    processData.step = &step;
    processData.macrCurr = &macrCPUCurrent;
    processData.macrOld = &macrCPUOld;
    
    // Number of devices
    checkCudaErrors(cudaGetDeviceCount(&info.numDevices));
    // if(N_GPUS > info.numDevices){
    //     printf("N_GPUS is higher than the number of detected GPUS\n");
    //     printf("N_GPUS: %d\n", N_GPUS);
    //     printf("Number of devices: %d\n", info.numDevices);
    //     return -1;
    // }
    info.numDevices = N_GPUS;

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    info.devices = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp)*N_GPUS);
    bcInfos = (BoundaryConditionsInfo*) malloc(sizeof(BoundaryConditionsInfo)*N_GPUS);
    gridsBC = (dim3*) malloc(sizeof(dim3)*N_GPUS);
    macrCPUCurrent.macrAllocation(IN_HOST);
    macrCPUOld.macrAllocation(IN_HOST);
    pop = (Populations*) malloc(sizeof(Populations) * N_GPUS);
    macr = (Macroscopics*) malloc(sizeof(Macroscopics) * N_GPUS);
    randomNumbers = (float**)malloc(sizeof(float*) * N_GPUS);
    /* ---------------------------------------------------------------------- */

    /* -------------- ALLOCATION AND CONFIGURATION FOR EACH GPU ------------- */
    // Streams for GPU
    cudaStream_t streamsLBM[N_GPUS];
    #ifdef IBM
    cudaStream_t streamsIBM[N_GPUS];
    #endif

    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        checkCudaErrors(cudaGetDeviceProperties(&(info.devices[i]), GPUS_TO_USE[i]));

        checkCudaErrors(cudaStreamCreate(&streamsLBM[i]));
        #ifdef IBM
        checkCudaErrors(cudaStreamCreate(&streamsIBM[i]));
        #endif
        #ifdef DENSITY_CORRECTION
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));  
        #endif

        pop[i].popAllocation();
        macr[i].macrAllocation(IN_VIRTUAL);
        if(RANDOM_NUMBERS)
        {
            checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[i], 
                sizeof(float)*NUMBER_LBM_NODES));
            initializationRandomNumbers(randomNumbers[i], CURAND_SEED);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("random numbers transfer error");
        }
    }
    getLastCudaError("LBM setup error");
    /* ---------------------------------------------------------------------- */

    /* ------------------ IBM ALLOCATION AND CONFIGURATION ------------------ */
    #ifdef IBM
    printf("-------------------------------- IBM INFORMATION -------------------------------\n");

    printf("Creating particles...\t"); fflush(stdout);
    createParticles(particles);
    printf("Particles created!\n"); fflush(stdout);

    particlesSoA.updateParticlesAsSoA(particles);
    ibmMacrsAux.ibmMacrsAuxAllocation();
    getLastCudaError("IBM setup error");

    ibmProcessData.step = &step;
    ibmProcessData.macrCurr = &macrCPUCurrent;
    #endif
    /* ---------------------------------------------------------------------- */

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 grid(((NX%N_THREADS)? (NX/N_THREADS+1) : (NX/N_THREADS)), NY, NZ);
    // threads in block
    dim3 threads(N_THREADS, 1, 1);

    // Grid and threads for memory transfers in multiGPUS
    dim3 gridTransfer(grid.x, grid.y, 1);
    dim3 threadsTransfer(N_THREADS, 1, 1);
    /* ---------------------------------------------------------------------- */

    /* ------------------------------- REPORT ------------------------------- */
    printSimInfo(&info);
    saveSimInfo(&info);
    /* ---------------------------------------------------------------------- */


    /* ----------------- BOUNDARY CONDITIONS INITIALIZATION ----------------- */
    // Divide in two fors to allow kernels of "gpuBuilBoundaryConditions"
    // to run in parallel. Otherwise they would run sequentially
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        gpuBuildBoundaryConditions<<<grid, threads>>>(pop[i].mapBC, i);
    }
    for (int i = 0; i < N_GPUS; i++) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        cudaDeviceSynchronize();
    }
    getLastCudaError("Initialization error");

    // Build auxiliary informations of boundary conditions for each GPU
    NodeTypeMap* hMapBC;
    checkCudaErrors(cudaMallocHost((void**)(&hMapBC), MEM_SIZE_MAP_BC));
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaMemcpy(hMapBC, pop[i].mapBC, MEM_SIZE_MAP_BC, cudaMemcpyDefault));
        bcInfos[i].setupBoundaryConditionsInfo(hMapBC);
    }
    cudaFreeHost(hMapBC);
    /* ---------------------------------------------------------------------- */

    /* ------------------------- LBM INITIALIZATION ------------------------- */
    // Load populations from files
    if(LOAD_CHECKPOINT)
    {
        loadSimCheckpoint(pop, macr, particlesSoA, &step);
    }
    else
    {
        step = INI_STEP;
        dim3 gridInit = grid;
        // Initialize ghost nodes
        gridInit.z += 1;
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            // Initialize populations
            gpuInitialization<<<gridInit, threads>>>(pop[i], macr[i], randomNumbers[i]);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        getLastCudaError("Initialization error");
    }
    int first_step = step;
    /* ---------------------------------------------------------------------- */
    #ifdef DENSITY_CORRECTION
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        h_mean_rho[i] = 0.0;
        checkCudaErrors(cudaMemcpy(d_mean_rho, h_mean_rho, sizeof(dfloat), cudaMemcpyHostToDevice)); 
        checkCudaErrors(cudaDeviceSynchronize());
    }
    #endif

    // Initialize Euler nodes for optimization
    #if IBM_EULER_OPTIMIZATION
    pEulerNodes.initializeEulerNodes(particlesSoA.pCenterArray);
    #endif

    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        size_t baseIdx = i*NUMBER_LBM_NODES;
        macrCPUCurrent.copyMacr(&macr[i], baseIdx, 0, false);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    macrCPUOld.copyMacr(&macrCPUCurrent, 0, 0, true);

    // Grid and thread definition for boundary conditions
    for(int i = 0; i < N_GPUS; i++)
        gridsBC[i] = dim3(((bcInfos[i].totalBCNodes%32)? (bcInfos[i].totalBCNodes/32+1) : 
                (bcInfos[i].totalBCNodes/32)), 1, 1); // TODO

    dim3 threadsBC(32, 1, 1);

    // Free random numbers
    if (RANDOM_NUMBERS) {
        for (int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            cudaFree(randomNumbers[i]);
        }
        free(randomNumbers);
    }

    // Timing
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    cudaEvent_t start, stop, start_step, stop_step;
    int last_step_sync = step;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));

    /* ------------------------------ LBM LOOP ------------------------------ */
    for(step = step; step < N_STEPS; step++)
    {
        int aux = step-INI_STEP;
        // WHAT NEEDS TO BE DONE IN THIS TIME STEP
        bool save = false, rep = false, repIBM = false, checkpoint = false, densityCorrection = false;
        if(aux != 0)
        {
            if(MACR_SAVE)
                save = !(aux % MACR_SAVE);
            if(DATA_REPORT)
                rep = !(aux % DATA_REPORT);
            if(CHECKPOINT_SAVE)
                checkpoint = !(aux % CHECKPOINT_SAVE);
            #ifdef IBM
            if(IBM_DATA_REPORT)
                repIBM = !(aux % IBM_DATA_REPORT);
            #endif
            #ifdef DENSITY_CORRECTION
                densityCorrection = true;
            #endif
        }
        // Save macroscopics to array in LBM kernel
        bool save_macr_to_array;
        #if defined(IBM) && !(IBM_EULER_OPTIMIZATION)
        save_macr_to_array = false;
        #else
        save_macr_to_array = rep || save || repIBM || ((step+1)>=(int)N_STEPS) || densityCorrection;
        #endif

        // LBM solver
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            gpuMacrCollisionStream<<<grid, threads>>>
                (pop[i].pop, pop[i].popAux, pop[i].mapBC, macr[i],
                save_macr_to_array,
                #ifdef DENSITY_CORRECTION
                d_mean_rho,
                #endif              
                step);
            //checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("LBM kernel error\n");

            #ifdef DENSITY_CORRECTION
                h_mean_rho[i] = mean_macro(macr[i], 0,step);
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
        }
        #ifdef DENSITY_CORRECTION
            if(N_GPUS > 1){
                for(int i = 1; i < N_GPUS; i++)
                    h_mean_rho[0] +=  h_mean_rho[i];   
            }
            h_mean_rho[0] = (h_mean_rho[0])/((dfloat)NUMBER_LBM_NODES*N_GPUS) - RHO_0;
            //printf("step %d rho_m %e \n ",step, h_mean_rho[0]);
            for(int i = 0; i < N_GPUS; i++){
                checkCudaErrors(cudaMemcpy(d_mean_rho, &h_mean_rho, sizeof(dfloat), cudaMemcpyHostToDevice)); 
            }
        #endif

        /*
        // While running kernel code, organize IBM Euler nodes
        #if defined(IBM) && IBM_EULER_OPTIMIZATION
        pEulerNodes.checkParticlesMovement();
        #endif
        */

        for(int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        // Populations ghost nodes transfer
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            int nxt = (i+1)%N_GPUS;
            gpuPopulationsTransfer<<<gridTransfer, threadsTransfer>>>
                (pop[i].popAux, pop[nxt].popAux);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("Mem transfer kernel error\n");
        }

        // Boundary conditions
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            if(bcInfos[i].totalBCNodes > 0){
                gpuApplyBC<<<gridsBC[i], threadsBC>>>
                    (pop[i].mapBC, pop[i].popAux, pop[i].pop, 
                    bcInfos[i].idxBCNodes, bcInfos[i].totalBCNodes, i);
            }
            getLastCudaError("BC kernel error\n");
        }

        // Synchronize and swap populations
        for (int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            checkCudaErrors(cudaDeviceSynchronize());
            pop[i].swapPop();
        }

        // IBM
        #ifdef IBM

        // Update particle nodes in each GPU
        if(!IBM_PARTICLE_UPDATE_INTERVAL || (step % IBM_PARTICLE_UPDATE_INTERVAL) == 0){
            particlesSoA.updateNodesGPUs();
            checkCudaErrors(cudaDeviceSynchronize());
        }

        #if IBM_EULER_OPTIMIZATION
        if(!IBM_EULER_UPDATE_INTERVAL || (step % IBM_EULER_UPDATE_INTERVAL) == 0)
        {
            pEulerNodes.checkParticlesMovement();
        }    
        #endif

        immersedBoundaryMethod(
            particlesSoA, macr, ibmMacrsAux, pop, grid, threads,
            streamsLBM, streamsIBM, step, 
            &pEulerNodes);

        // Save particles informations
        if(IBM_PARTICLES_SAVE && !(step % IBM_PARTICLES_SAVE)){
            saveParticlesInfo(particlesSoA, step, IBM_PARTICLES_NODES_SAVE);
        }
        #endif

        // Synchronizing data (macroscopics) between GPU and CPU
        if(save || rep || repIBM)
        {
            printf("\n------------------------- Synchronizing in step %06d -------------------------\n", step);
            fflush(stdout);

            // Timing between syncs
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
            checkCudaErrors(cudaEventRecord(stop_step, 0));
            checkCudaErrors(cudaEventSynchronize(stop_step));
            float elapsedTime;
            checkCudaErrors(cudaEventElapsedTime(&(elapsedTime), start_step, stop_step));
            
            elapsedTime *= 0.001;
            // Calculate MLUPS
            size_t nodesUpdatedSync = (step-last_step_sync) * NUMBER_LBM_NODES * N_GPUS;
            info.MLUPS = (nodesUpdatedSync / 1e6) / elapsedTime;
            info.timeElapsed += elapsedTime;
            last_step_sync = step;
            // Save simulation info
            saveSimInfo(&info);
            
            printf("                  MLUPS: %f\n", info.MLUPS);
            printf("       Elapsed time (s): %f\n", info.timeElapsed);
            fflush(stdout);

            // Restart start and stop event
            checkCudaErrors(cudaEventDestroy(start_step));
            checkCudaErrors(cudaEventCreate(&start_step));
            checkCudaErrors(cudaEventDestroy(stop_step));
            checkCudaErrors(cudaEventCreate(&stop_step));

            checkCudaErrors(cudaEventRecord(start_step, 0));

            if(rep)
                macrCPUOld.copyMacr(&macrCPUCurrent, 0, 0, true);
            for(int i = 0; i < N_GPUS; i++){
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
                macrCPUCurrent.copyMacr(&macr[i], NUMBER_LBM_NODES*i);
                checkCudaErrors(cudaDeviceSynchronize());
            }

            // for(int z = 0; z < NZ_TOTAL; z++)
            // {
            //     for(int y = 0; y < NY; y++)
            //     {
            //         for(int x = 0; x < NX; x++)
            //         {
            //             size_t idx = idxScalar(x, y, z);
            //             size_t idxGPU = idxScalar(x, y, z+MACR_BORDER_NODES);
            //             printf("(z, y, x) %d %d %d (rho cpu, gpu) %.2f, %.2f\n", z, y, x, macrCPUCurrent.rho[idx], macr[0].rho[idxGPU]);
            //         }
            //     }
            // }
        }

        if(checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
            fflush(stdout);
            saveSimCheckpoint(pop, macr, particlesSoA, &step);
            // Save info as well (to know when it stopped, conf, etc.)
            saveSimInfo(&info);
        }
        // Save macroscopics
        if(save)
        {
            printf("\n---------------------------- Saving in step %06d -----------------------------\n", step); 
            fflush(stdout);
            saveAllMacrBin(&macrCPUCurrent, step);
        }

        // Report data
        if(rep)
        {
            treatData(&processData,step);
            printTreatData(&processData); 
            fflush(stdout);
            if(DATA_SAVE)
            {
                saveTreatData(&processData);
            }
            if(DATA_STOP)
            {
                if(stopSim(&processData))
                {
                    printf("Stopping because of LBM\n");
                    break;
                }
            }
        }

        // Report IBM data
        #ifdef IBM
        if(repIBM){
            treatDataIBM(&ibmProcessData, particlesSoA);
            printTreatDataIBM(&ibmProcessData);
            fflush(stdout);
            if(IBM_DATA_SAVE)
            {
                saveTreatDataIBM(&ibmProcessData);
            }
            if(IBM_DATA_STOP)
            {
                if(stopSimIBM(&ibmProcessData, particlesSoA))
                {
                    printf("Stopping because of IBM\n");
                    break;
                }
            }
        }
        #endif
    }
    /* ---------------------------------------------------------------------- */

    // Timing
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&(info.timeElapsed), start, stop));

    checkCudaErrors(cudaEventDestroy(start_step));
    checkCudaErrors(cudaEventDestroy(stop_step));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    info.timeElapsed *= 0.001;

    // Save final macroscopics
    for(int i = 0; i < N_GPUS; i++){
        macrCPUCurrent.copyMacr(&macr[i], NUMBER_LBM_NODES*i);
    }
    saveAllMacrBin(&macrCPUCurrent, step);
    checkCudaErrors(cudaDeviceSynchronize());

    // Save final IBM values
    #ifdef IBM
    saveParticlesInfo(particlesSoA, step, IBM_PARTICLES_NODES_SAVE);
    if(IBM_DATA_SAVE){
        saveTreatDataIBM(&ibmProcessData);
    }
    #endif

    // Evaluate performance
    info.totalSteps = step - first_step;
    size_t nodesUpdated = info.totalSteps * NUMBER_LBM_NODES * N_GPUS;
    info.MLUPS = (nodesUpdated / 1e6) / info.timeElapsed;
    // bandwidth for AB scheme and does not consider macroscopics transfers
    info.bandwidth = MEM_SIZE_POP*2.0*N_GPUS / (info.timeElapsed*BYTES_PER_GB) 
        * info.totalSteps;

    // Save last checkpoint, if required
    if(CHECKPOINT_SAVE != 0)
            saveSimCheckpoint(pop, macr, particlesSoA, &step);
    // Save simulation info
    saveSimInfo(&info);

    // Report data (last calculated one)
    if(DATA_REPORT)
    {
        printTreatData(&processData);
        if(DATA_SAVE)
            saveTreatData(&processData);
    }
    printSimInfo(&info);

    /* ---------------------------- FREE MEMORY ----------------------------- */
    // Free memory for each GPU
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        checkCudaErrors(cudaStreamDestroy(streamsLBM[i]));
        #ifdef IBM
        checkCudaErrors(cudaStreamDestroy(streamsIBM[i]));
        #endif
        pop[i].popFree();
        macr[i].macrFree();
        bcInfos[i].freeIdxBC();
        
        #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho);
        #endif

    }

    // Free CPU variables
    free(pop);
    free(macr);
    macrCPUCurrent.macrFree();
    macrCPUOld.macrFree();
    free(info.devices);
    free(bcInfos);
    free(gridsBC);

    #ifdef IBM
    freeIBMProc(&ibmProcessData);
    for(int i = 0; i < NUM_PARTICLES; i++){
        free(particles[i].nodes);
    }
    free(particles);
    particlesSoA.freeNodesAndCenters();
    ibmMacrsAux.ibmMacrsAuxFree();
    #if IBM_EULER_OPTIMIZATION
    pEulerNodes.freeEulerNodes();
    #endif
    #endif
    /* ---------------------------------------------------------------------- */

    fflush(stdout);

    return 0;
}