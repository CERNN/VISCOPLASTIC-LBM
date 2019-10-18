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


int main()
{
    // VARIABLE DECLARATIONS
    Populations* pop;
    Macroscopics* macr;
    Macroscopics macrCPUCurrent;
    Macroscopics macrCPUOld;
    MacrProc processData;
    BoundaryConditionsInfo bcInfo;
    SimInfo info;
    int step = INI_STEP;

    // SETUP SABING FOLDER
    folderSetup();

    // INITALIZE PROCESS DATA
    processData.step = &step;
    processData.macrCurr = &macrCPUCurrent;
    processData.macrOld = &macrCPUOld;
    
    // NUMBER OF DEVICES
    checkCudaErrors(cudaGetDeviceCount(&info.numDevices));
    const int N_GPUS = 1;

    // ALLOCATION FOR CPU
    info.devices = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp)*N_GPUS);
    macrCPUCurrent.macrAllocation(IN_HOST);
    macrCPUOld.macrAllocation(IN_HOST);

    // STREAMS AND MEMORY ALLOCATION FOR GPU
    cudaStream_t streamsKernelLBM[1]; // stream kernel for each GPU
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMallocManaged((void**)&pop, sizeof(Populations)*N_GPUS));
    checkCudaErrors(cudaMallocManaged((void**)&macr, sizeof(Macroscopics)*N_GPUS));

    // ALLOCATION AND CONFIGURATION FOR EACH GPU
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaGetDeviceProperties(&(info.devices[i]), i));
        checkCudaErrors(cudaStreamCreate(&streamsKernelLBM[i]));
        pop[i].popAllocation();
        macr[i].macrAllocation(IN_VIRTUAL);
    }
    

/*  
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    ------------------ CODE BELOW DOES NOT SUPPORT MULTI GPU! -----------------
    ----------------------------- MUST BE UPDATED! ----------------------------
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
*/

    // GRID AND THREADS DEFINITION
    dim3 grid(((NX%nThreads)? (NX/nThreads+1) : (NX/nThreads)), NY, NZ);
    // threads in block
    dim3 threads(nThreads, 1, 1);

    // REPORT
    printParamInfo(&info, true); fflush(stdout);
    printGPUInfo(&info); fflush(stdout);

    // BOUNDARY CONDITIONS INITIALIZATION
    gpuBuildBoundaryConditions<<<grid, threads>>>(pop[0].mapBC);
    checkCudaErrors(cudaDeviceSynchronize());
    bcInfo.setupBoundaryConditionsInfo(pop->mapBC);

    // LBM INITIALIZATION
    if(LOAD_POP)
    {
        FILE* filePop = fopen(STR_POP, "r");
        initializationPop(&pop[0], filePop);
        gpuUpdateMacr<<<grid, threads>>>(&pop[0], &macr[0]);
    }
    else 
    {
        if(LOAD_MACR)
        {   
            FILE* fileRho = fopen(STR_RHO, "r");
            FILE* fileUx = fopen(STR_UX, "r");
            FILE* fileUy = fopen(STR_UY, "r");
            FILE* fileUz = fopen(STR_UZ, "r");
            initializationMacr(&macr[0], fileRho, fileUx, fileUy, fileUz);
        }
        gpuInitialization<<<grid, threads>>>(&pop[0], &macr[0], LOAD_MACR);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // TIMING
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    
    // LBM
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

        // LBM SOLVER
        gpuBCMacrCollisionStream<<<grid, threads, 0, streamsKernelLBM[0]>>>
            (pop->pop, pop->popAux, pop->mapBC, 
            &macr[0], rep || save || ((step+1)>=(int)N_STEPS), step);
        getLastCudaError("lbm kernel error");
        
        // if there are non local boundary conditions
        if(bcInfo.hasNonLocalBC())
        {
            dim3 threadsBC(32,1,1);
            dim3 gridBC(bcInfo.totalNonLocalBCNodes/32, 1, 1);
            if(bcInfo.totalNonLocalBCNodes%32)
                gridBC.x++;
            
            checkCudaErrors(cudaStreamSynchronize(streamsKernelLBM[0]));
            // applies to pop->pop (auxiliary populations) non local boundary conditions
            gpuApplyNonLocalBC<<<gridBC, threadsBC, 0, streamsKernelLBM[0]>>>
                (pop->popAux, pop->mapBC, pop->pop, 
                bcInfo.idxNonLocalBCNodes, bcInfo.totalNonLocalBCNodes);
            getLastCudaError("application of non local boundary conditions error");

            checkCudaErrors(cudaStreamSynchronize(streamsKernelLBM[0]));
            // synchronizes the boundary conditions applied to pop->popAux (valid populations)
            gpuSynchronizeNonLocalBC<<<gridBC, threadsBC, 0, streamsKernelLBM[0]>>>
                (pop->popAux, pop->pop, 
                bcInfo.idxNonLocalBCNodes, bcInfo.totalNonLocalBCNodes);
            getLastCudaError("synchronization of non local boundary conditions error");
        }

        checkCudaErrors(cudaStreamSynchronize(streamsKernelLBM[0]));
        pop[0].swapPop();

        // SYNCHRONIZING
        if(save || rep)
        {
            printf("\n------------------------- Synchronizing in step %06d -------------------------\n", step);
            checkCudaErrors(cudaStreamSynchronize(streamsKernelLBM[0]));
            
            macrCPUOld.copyMacr(&macrCPUCurrent);
            macrCPUCurrent.copyMacr(&macr[0]); 
        }

        // SAVE
        if(save)
        {
            printf("\n---------------------------- Saving in step %06d -----------------------------\n", step);
            saveAllMacrBin(&macrCPUCurrent, step);
        }

        // REPORT
        if(rep)
        {
            treatData(&processData);
            printTreatData(&processData);
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

    // TIMING
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&(info.timeElapsed), start, stop));
    info.timeElapsed *= 0.001;

    // SAVE FINAL MACROSCOPICS
    macrCPUCurrent.copyMacr(&macr[0]); 
    saveAllMacrBin(&macrCPUCurrent, step);

    // SAVE FINAL POPULATIONS (IF REQUIRED)
    if(POP_SAVE)
        savePopBin(pop, step);

    // EVALUATE PERFORMANCE
    info.totalSteps = step - INI_STEP;
    size_t nodesUpdated = info.totalSteps * numberNodes;
    info.MLUPS = (nodesUpdated / 1e6) / info.timeElapsed;
    // bandwidth for AB scheme and does not consider macroscopics transfers
    info.bandwidth = memSizePop * 2.0 / (info.timeElapsed*BYTES_PER_GB) * info.totalSteps;

    // SIMULATION INFO
    saveSimInfo(&info);

    // REPORT
    if(DATA_REPORT)
    {
        printTreatData(&processData);
        if(DATA_SAVE)
            saveTreatData(&processData);
    }
    printParamInfo(&info, true);
    printGPUInfo(&info);
    
    // FREE MEMORY FOR EACH GPU
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamDestroy(streamsKernelLBM[i]));
        pop[i].popFree();
        macr[i].macrFree();
    }
    bcInfo.freeIdxNonLocal();

    // FREE GPU VARIABLES
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaFree(pop));
    checkCudaErrors(cudaFree(macr));

    // FREE CPU VARIABLES
    macrCPUCurrent.macrFree();
    macrCPUOld.macrFree();
    free(info.devices);

    return 0;
}