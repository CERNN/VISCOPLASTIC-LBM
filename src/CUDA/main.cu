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

#ifdef IBM
#include "IBM/ibm.h"
#endif // IBM


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
    cudaStream_t* streamsKernelLBM = (cudaStream_t*) malloc(sizeof(cudaStream_t)*N_GPUS); 
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaGetDeviceProperties(&(info.devices[i]), i));
        checkCudaErrors(cudaStreamCreate(&streamsKernelLBM[i]));

        pop[i]. popAllocation();
        macr[i].macrAllocation(IN_VIRTUAL);
        if(RANDOM_NUMBERS)
        {
            checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[i], 
                sizeof(float)*numberNodes));
            initializationRandomNumbers(randomNumbers[i], CURAND_SEED);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("random numbers transfer error");
        }
    }
    /* ---------------------------------------------------------------------- */

    /* -------------------------------- IBM --------------------------------- */

    /* ---------------------------------------------------------------------- */

    // ------------------ GRID AND THREADS DEFINITION FOR LBM ------------------
    dim3 grid(((NX%nThreads)? (NX/nThreads+1) : (NX/nThreads)), NY, NZ);
    // threads in block
    dim3 threads(nThreads, 1, 1);

    // Grid and threads for memory transfers in multiGPUS
    dim3 gridTransfer(grid.x, grid.y, 1);
    dim3 threadsTransfer(nThreads, 1, 1);
    /* ---------------------------------------------------------------------- */

    /* ------------------------------- REPORT ------------------------------- */
    printParamInfo(&info, true); fflush(stdout);
    printGPUInfo(&info); fflush(stdout);
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
    checkCudaErrors(cudaMallocHost((void**)(&hMapBC), memSizeMapBC));
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaMemcpy(hMapBC, pop[i].mapBC, memSizeMapBC, cudaMemcpyDefault));
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
            if(fileRho == nullptr || fileUz == nullptr 
                || fileUy == nullptr || fileUx == nullptr)
            {
                printf("Error reading macroscopics files\n");
                return -1;
            }
            // Load macroscopics from files
            initializationMacr(&macrCPUCurrent, fileRho, fileUx, fileUy, fileUz);
            fclose (fileRho);
            fclose (fileUx);
            fclose (fileUy);
            fclose (fileUz);
        }
        
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            // Copy macroscopics to GPU if required
            if(LOAD_MACR){
                size_t baseIdx = i*numberNodes;
                macr[i].copyMacr(&macrCPUCurrent, 0, baseIdx, false);
                checkCudaErrors(cudaDeviceSynchronize());
            }
            // Initialize populations
            gpuInitializationPop<<<grid, threads>>>(pop[i], macr[i], LOAD_MACR, randomNumbers[i]);
            // checkCudaErrors(cudaDeviceSynchronize());
        }
        getLastCudaError("Initialization error");
    }
    /* ---------------------------------------------------------------------- */

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
        bool save = false, rep = false;
        if(aux != 0)
        {
            if(MACR_SAVE != 0)
                save = !(aux % MACR_SAVE);
            if(DATA_REPORT != 0)
                rep = !(aux % DATA_REPORT);
        }

        // LBM solver
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(i));
            gpuMacrCollisionStream<<<grid, threads>>>
                (pop[i].pop, pop[i].popAux, pop[i].mapBC, 
                macr[i], rep || save || ((step+1)>=(int)N_STEPS), step);
            //checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("LBM kernel error\n");
        }

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

        // Synchronizing data (macroscopics) between GPU and CPU
        if(save || rep)
        {
            printf("\n------------------------- Synchronizing in step %06d -------------------------\n", step); 
            fflush(stdout);

            macrCPUOld.copyMacr(&macrCPUCurrent, 0, 0, true);
            for(int i = 0; i < N_GPUS; i++){
                macrCPUCurrent.copyMacr(&macr[i], numberNodes*i);
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
                    break;
            }
        }
    }
    /* ---------------------------------------------------------------------- */

    // Timing
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&(info.timeElapsed), start, stop));
    info.timeElapsed *= 0.001;

    // Save final macroscopics
    for(int i = 0; i < N_GPUS; i++){
        macrCPUCurrent.copyMacr(&macr[i], numberNodes*i);
    } 
    saveAllMacrBin(&macrCPUCurrent, step);

    // Save final populations (if required)
    if(POP_SAVE)
        savePopBin(pop, step);

    // Evaluate performance
    info.totalSteps = step - INI_STEP;
    size_t nodesUpdated = info.totalSteps * numberNodes * N_GPUS;
    info.MLUPS = (nodesUpdated / 1e6) / info.timeElapsed;
    // bandwidth for AB scheme and does not consider macroscopics transfers
    info.bandwidth = memSizePop*2.0*N_GPUS / (info.timeElapsed*BYTES_PER_GB) 
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
    printParamInfo(&info, true);
    printGPUInfo(&info);
    fflush(stdout);

    /* ---------------------------- FREE MEMORY ----------------------------- */
    // Free memory for each GPU
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamDestroy(streamsKernelLBM[i]));
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
    /* ---------------------------------------------------------------------- */

    return 0;
}