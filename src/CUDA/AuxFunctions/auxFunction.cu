#include "auxFunction.cuh"

__host__
dfloat mean_macro(Macroscopics const macr, int macro_index, size_t step)
{
    const dim3 grid(((NX% N_THREADS) ? (NX / N_THREADS + 1) : (NX / N_THREADS)), NY, NZ);
    const dim3 threads(N_THREADS, 1, 1);


    dfloat* sum;
    cudaMalloc((void**)&sum, NUM_BLOCK * sizeof(dfloat));

    int nt_x = N_THREADS;
    int nt_y = 1;
    int nt_z = 1;

    int nb_x = NX / nt_x;
    int nb_y = NY;
    int nb_z = NZ;

    int current_block_size = nb_x * nb_y * nb_z;

    reductionMacro << <grid, threads >> > (macr.rho, sum);

   
while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) {
            reductionArray << <1, dim3(nb_x, 1, 1) >> > (sum, sum);

            cudaDeviceSynchronize();
            break;
        }
        else {
            nb_x = (current_block_size < nt_x ? 1 : current_block_size / nt_x);
            nb_y = 1;
            nb_z = 1;
            current_block_size = nb_x * nb_y * nb_z;

            reductionArray << <dim3(nb_x, 1, 1), dim3(nt_x, 1, 1) >> > (sum, sum);

            cudaDeviceSynchronize();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sum, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    cudaFree(sum);
    
    return temp;

    /*
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(meanMacro, sum, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    printf("step %d rho_m %e \n ",step, meanMacro);
    cudaFree(sum);*/
    
}


