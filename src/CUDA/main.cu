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
#include "boundaryConditionsBuilder.h"
#include "structs/boundaryConditionsInfo.h"

#include "IBM/ibm.h"
#include "IBM/ibmParticlesCreation.h"
#include "IBM/ibmTreatData.h"


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

    #ifdef IBM
    Particle particles[NUM_PARTICLES];
    ParticlesSoA particlesSoA;
    dfloat3SoA velAuxIBM[N_GPUS];
    ParticleEulerNodesUpdate pEulerNodes;

    IBMProc ibmProcessData;
    allocateIBMProc(&ibmProcessData);
    #endif

    // Setup saving folder
    folderSetup();

    // Initializes process data
    processData.step = &step;
    processData.macrCurr = &macrCPUCurrent;
    processData.macrOld = &macrCPUOld;
    
    // Number of devices
    checkCudaErrors(cudaGetDeviceCount(&info.numDevices));
    if(N_GPUS > info.numDevices){
        printf("N_GPUS is higher than the number of detected GPUS\n");
        printf("N_GPUS: %d\n", N_GPUS);
        printf("Number of devices: %d\n", info.numDevices);
        return -1;
    }
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
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaGetDeviceProperties(&(info.devices[i]), i));

        checkCudaErrors(cudaStreamCreate(&streamsLBM[i]));
        #ifdef IBM
        checkCudaErrors(cudaStreamCreate(&streamsIBM[i]));
        
        velAuxIBM[i].allocateMemory(NUMBER_LBM_NODES*sizeof(dfloat));
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
    /* ---------------------------------------------------------------------- */

    /* ------------------ IBM ALLOCATION AND CONFIGURATION ------------------ */
    #ifdef IBM
    printf("-------------------------------- IBM INFORMATION -------------------------------\n");

    printf("Creating particles...\t"); fflush(stdout);
    createParticles(particles);
    printf("Particles created!\n"); fflush(stdout);

    particlesSoA.updateParticlesAsSoA(particles);
    #if IBM_EULER_OPTIMIZATION
    pEulerNodes.initializeEulerNodes(particlesSoA.pCenterArray);
    #endif
    const unsigned int threadsIBM = 64;
    const unsigned int pNumNodes = particlesSoA.nodesSoA.numNodes;
    const unsigned int gridIBM = pNumNodes % threadsIBM ? pNumNodes / threadsIBM + 1 : pNumNodes / threadsIBM;
    
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
        checkCudaErrors(cudaSetDevice(i));
        gpuBuildBoundaryConditions<<<grid, threads>>>(pop[i].mapBC, i);
        getLastCudaError("Initialization error");
    }
    for (int i = 0; i < N_GPUS; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceSynchronize();
    }

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
    if(LOAD_POP)
    {
        FILE* filePop = fopen(STR_POP, "rb");
        FILE* filePopAux = fopen(STR_POP_AUX, "rb");
        if(filePop == nullptr || filePopAux == nullptr)
        {
            printf("Error reading population file\n");
            return -1;
        }
        initializationPop(pop, filePop, filePopAux);
        fclose (filePop);
        fclose (filePopAux);

        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            gpuUpdateMacr<<<grid, threads>>>(pop[i], macr[i]);
            getLastCudaError("Update macroscopics error");
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    else 
    {
        // Load macroscopics from files
        if(LOAD_MACR)
        {
            FILE* fileRho = fopen(STR_RHO, "rb");
            FILE* fileUx = fopen(STR_UX, "rb");
            FILE* fileUy = fopen(STR_UY, "rb");
            FILE* fileUz = fopen(STR_UZ, "rb");
            FILE* fileFx = fopen(STR_FX, "rb");
            FILE* fileFy = fopen(STR_FY, "rb");
            FILE* fileFz = fopen(STR_FZ, "rb");
            FILE* fileOmega = fopen(STR_OMEGA, "rb");

            if(fileRho == nullptr || fileUz == nullptr 
                || fileUy == nullptr || fileUx == nullptr
                #ifdef IBM
                || fileFx == nullptr || fileFy == nullptr || fileFz == nullptr
                #endif
                #ifdef NON_NEWTONIAN_FLUID
                || fileOmega == nullptr
                #endif
            ){
                printf("Error reading macroscopics files. (1 for not found):\n");
                printf("FILE_RHO=%d; FILE_UX=%d; FILE_UY=%d; FILE_UZ=%d;\n", 
                    fileRho==nullptr, fileUx==nullptr, fileUy==nullptr, fileUz==nullptr);
                #ifdef IBM
                printf("FILE_FX=%d; FILE_FY=%d; FILE_FZ=%d;\n", 
                    fileFx==nullptr, fileFy==nullptr, fileFz==nullptr);
                #endif
                #ifdef NON_NEWTONIAN_FLUID
                printf("FILE_OMEGA=%d\n", fileOmega==nullptr);
                #endif
                return -1;
            }
            // Load macroscopics from files
            initializationMacr(&macrCPUCurrent, fileRho, fileUx, fileUy, fileUz, 
                fileFx, fileFy, fileFz, fileOmega);
            fclose (fileRho);
            fclose (fileUx);
            fclose (fileUy);
            fclose (fileUz);
            #ifdef IBM
            fclose (fileFx);
            fclose (fileFy);
            fclose (fileFz);
            #endif
            #ifdef NON_NEWTONIAN_FLUID
            fclose (fileOmega);
            #endif
        }
        
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            // Copy macroscopics to GPU if required
            if(LOAD_MACR){
                size_t baseIdx = i*NUMBER_LBM_NODES;
                macr[i].copyMacr(&macrCPUCurrent, 0, baseIdx, false);
                checkCudaErrors(cudaDeviceSynchronize());
            }
            // Initialize populations
            gpuInitialization<<<grid, threads>>>(pop[i], macr[i], LOAD_MACR, randomNumbers[i]);
            // checkCudaErrors(cudaDeviceSynchronize());
        }
        getLastCudaError("Initialization error");
    }
    /* ---------------------------------------------------------------------- */

    if(!LOAD_MACR){
        for(int i = 0; i < N_GPUS; i++){
            size_t baseIdx = i*NUMBER_LBM_NODES;
            macrCPUCurrent.copyMacr(&macr[i], baseIdx, 0, false);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        macrCPUOld.copyMacr(&macrCPUCurrent, 0, 0, true);
    }

    // Grid and thread definition for boundary conditions
    for(int i = 0; i < N_GPUS; i++)
        gridsBC[i] = dim3(((bcInfos[i].totalBCNodes%32)? (bcInfos[i].totalBCNodes/32+1) : 
                (bcInfos[i].totalBCNodes/32)), 1, 1); // TODO

    dim3 threadsBC(32, 1, 1);

    // Free random numbers
    if (RANDOM_NUMBERS) {
        for (int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(i));
            cudaFree(randomNumbers[i]);
        }
        free(randomNumbers);
    }

    // Timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    /* ------------------------------ LBM LOOP ------------------------------ */
    for(step = INI_STEP; step < N_STEPS; step++)
    {
        int aux = step-INI_STEP;
        // WHAT NEEDS TO BE DONE IN THIS TIME STEP
        bool save = false, rep = false, repIBM = false;
        if(aux != 0)
        {
            if(MACR_SAVE != 0)
                save = !(aux % MACR_SAVE);
            if(DATA_REPORT != 0)
                rep = !(aux % DATA_REPORT);
            #ifdef IBM
            if(IBM_DATA_REPORT != 0)
                repIBM = !(aux % IBM_DATA_REPORT);
            #endif
        }
        // Save macroscopics to array in LBM kernel
        bool save_macr_to_array;
        #ifdef IBM
        save_macr_to_array = false;
        #else
        save_macr_to_array = rep || save || repIBM || ((step+1)>=(int)N_STEPS);
        #endif

        // LBM solver
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            gpuMacrCollisionStream<<<grid, threads>>>
                (pop[i].pop, pop[i].popAux, pop[i].mapBC, macr[i],
                save_macr_to_array, step);
            //checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("LBM kernel error\n");
        }

        // While running kernel code, organize IBM Euler nodes
        #if defined(IBM) && IBM_EULER_OPTIMIZATION
        pEulerNodes.checkParticlesMovement();
        #endif

        for(int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        
        if(N_GPUS > 1) {
            // Populations transfer
            for(int i = 0; i < N_GPUS; i++){
                checkCudaErrors(cudaSetDevice(i));
                int nxt = (i+1)%N_GPUS;
                gpuPopulationsTransfer<<<gridTransfer, threadsTransfer>>>
                    (pop[i].pop, pop[i].popAux, pop[nxt].pop, pop[nxt].popAux);
                checkCudaErrors(cudaDeviceSynchronize());
                getLastCudaError("Mem transfer kernel error\n");
            }
        }

        // Boundary conditions
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            if(bcInfos[i].totalBCNodes > 0){
                gpuApplyBC<<<gridsBC[i], threadsBC>>>
                    (pop[i].mapBC, pop[i].popAux, pop[i].pop, 
                    bcInfos[i].idxBCNodes, bcInfos[i].totalBCNodes);
            }
            getLastCudaError("BC kernel error\n");
        }

        // Synchronize and swap populations
        for (int i = 0; i < N_GPUS; i++) {
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaDeviceSynchronize());
            pop[i].swapPop();
        }

        // IBM
        #ifdef IBM
        immersedBoundaryMethod(
            particlesSoA, macr, velAuxIBM, pop, grid, threads,
            gridIBM, threadsIBM, streamsLBM, streamsIBM, step, 
            &pEulerNodes);

        // Save particles informations
        if(IBM_PARTICLES_SAVE != 0 && !(step % IBM_PARTICLES_SAVE)){
            saveParticlesInfo(particlesSoA, step, IBM_PARTICLES_NODES_SAVE);
        }
        #endif

        // Synchronizing data (macroscopics) between GPU and CPU
        if(save || rep || repIBM)
        {
            printf("\n------------------------- Synchronizing in step %06d -------------------------\n", step); 
            fflush(stdout);

            if(rep)
                macrCPUOld.copyMacr(&macrCPUCurrent, 0, 0, true);
            for(int i = 0; i < N_GPUS; i++){
                checkCudaErrors(cudaSetDevice(i));
                macrCPUCurrent.copyMacr(&macr[i], NUMBER_LBM_NODES*i);
                checkCudaErrors(cudaDeviceSynchronize());
            }
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
            treatData(&processData);
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
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&(info.timeElapsed), start, stop));

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

    // Save final populations (if required)
    if(POP_SAVE)
        savePopBin(pop, step);

    // Evaluate performance
    info.totalSteps = step - INI_STEP;
    size_t nodesUpdated = info.totalSteps * NUMBER_LBM_NODES * N_GPUS;
    info.MLUPS = (nodesUpdated / 1e6) / info.timeElapsed;
    // bandwidth for AB scheme and does not consider macroscopics transfers
    info.bandwidth = MEM_SIZE_POP*2.0*N_GPUS / (info.timeElapsed*BYTES_PER_GB) 
        * info.totalSteps;

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
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamDestroy(streamsLBM[i]));
        #ifdef IBM
        checkCudaErrors(cudaStreamDestroy(streamsIBM[i]));
        velAuxIBM[i].freeMemory();
        #endif
        pop[i].popFree();
        macr[i].macrFree();
        bcInfos[i].freeIdxBC();
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
    #endif
    /* ---------------------------------------------------------------------- */

    fflush(stdout);

    return 0;
}