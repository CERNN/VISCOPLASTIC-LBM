#include "lbm_d3q19.cuh"

__host__
void pop_initialisation(dfloat * f1_gpu, FILE* file_pop, dfloat * f2_gpu, dfloat * rho_gpu,
    dfloat * ux_gpu, dfloat * uy_gpu, dfloat * uz_gpu)
{
    dfloat* tmp = (dfloat*)malloc(mem_size_pop);
    if (file_pop != NULL)
    {
        fread(tmp, mem_size_pop, 1, file_pop);
        checkCudaErrors(cudaMemcpy(f1_gpu, tmp, mem_size_pop, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(f2_gpu, tmp, mem_size_pop, cudaMemcpyHostToDevice));
    }
    delete(tmp);

    update_rho_u(f1_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
}


__host__
void macr_initialisation(dfloat * f1_gpu, dfloat * f2_gpu, dfloat * rho_gpu, FILE * file_rho,
    dfloat * ux_gpu, FILE * file_ux, dfloat * uy_gpu, FILE * file_uy, dfloat * uz_gpu, FILE * file_uz)
{
    dfloat* tmp = (dfloat*)malloc(mem_size_scalar);
    if (file_rho != NULL)
    {
        fread(tmp, mem_size_scalar, 1, file_rho);
        checkCudaErrors(cudaMemcpy(rho_gpu, tmp, mem_size_scalar, cudaMemcpyHostToDevice));
    }
    if (file_ux != NULL)
    {
        fread(tmp, mem_size_scalar, 1, file_ux);
        checkCudaErrors(cudaMemcpy(ux_gpu, tmp, mem_size_scalar, cudaMemcpyHostToDevice));
    }
    if (file_uy != NULL)
    {
        fread(tmp, mem_size_scalar, 1, file_uy);
        checkCudaErrors(cudaMemcpy(uy_gpu, tmp, mem_size_scalar, cudaMemcpyHostToDevice));
    }
    if (file_uz != NULL)
    {
        fread(tmp, mem_size_scalar, 1, file_uz);
        checkCudaErrors(cudaMemcpy(uz_gpu, tmp, mem_size_scalar, cudaMemcpyHostToDevice));
    }
    delete(tmp);
    initialisation(f1_gpu, f2_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, true, false);
}


__host__
void initialisation(
    dfloat* f1, 
    dfloat* f2, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz,
    bool is_macr_init,
    bool is_taylor_green)
{
    // blocks in grid
    dim3 grid(N_X / nThreads_X, N_Y / nThreads_Y, N_Z / nThreads_Z);
    // threads in block
    dim3 threads(nThreads_X, nThreads_Y, nThreads_Z);
    
    if(!is_taylor_green)
        gpu_initialisation <<<grid, threads >>> (f1, f2, rho, ux, uy, uz, is_macr_init);
    else
        gpu_taylor_green_vortex_initialisation <<<grid, threads >>> (f1, f2, rho, ux, uy, uz);
    getLastCudaError("initialisation error");
}


__global__
void gpu_initialisation(
    dfloat* f, 
    dfloat* f_post, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz,
    bool is_macr_init)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= N_X || y >= N_Y || z >= N_Z)
        return;
    size_t index = index_scalar_d3(x, y, z);

    for (int i = 0; i < Q; i++)
    {
        if (!is_macr_init)
        {
            f[index_pop_d3q19(x, y, z, i)] = RHO_0 * w[i];
            f_post[index_pop_d3q19(x, y, z, i)] = RHO_0 * w[i];
        }
        else
        {
            dfloat feq = gpu_f_eq(w[i] * rho[index],
                3 * (ux[index] * c_x[i] + uy[index] * c_y[i] + uz[index] * c_z[i]),
                1 - (ux[index] * ux[index] + uy[index] * uy[index] + uz[index] * uz[index]));

            f[index_pop_d3q19(x, y, z, i)] = feq;
            f_post[index_pop_d3q19(x, y, z, i)] = feq;
        }
    }

    if (!is_macr_init)
    {
        ux[index] = 0;
        uy[index] = 0;
        uz[index] = 0;
        rho[index] = RHO_0;
    }
}

__global__ 
void gpu_taylor_green_vortex_initialisation(
    dfloat* f, 
    dfloat* f_post, 
    dfloat* rho, 
    dfloat* ux, 
    dfloat* uy, 
    dfloat* uz)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= N_X || y >= N_Y || z >= N_Z)
        return;
    size_t index = index_scalar_d3(x, y, z);

    dfloat L = 2.0*M_PI/N_X;
    // macroscopic initial
    rho[index] = RHO_0 + (3.0/16.0)*RHO_0*U_MAX*U_MAX*(cos(2*(x+0.5) / L) + cos(2*(y+0.5) / L))*(cos(2*(z+0.5) / L) + 2.0);
    ux[index] = U_MAX * sin((x+0.5) / L)*cos((y+0.5) / L) * cos((z+0.5) / L);
    uy[index] = -U_MAX * cos((x+0.5) / L)*sin((y+0.5) / L) * cos((z+0.5) / L);
    uz[index] = 0.0;
    for (int i = 0; i < Q; i++)
    {

        dfloat feq, f_neq;
        dfloat Axx, Ayy, Axy, Ayx, Axz, Ayz;
        Axx = + (c_x[i]*c_x[i] - 1.0/3.0)*cos((x+0.5)/L)*cos((y+0.5)/L)*cos((z+0.5)/L);
		Ayy = - (c_y[i]*c_y[i] - 1.0/3.0)*cos((x+0.5)/L)*cos((y+0.5)/L)*cos((z+0.5)/L);
		Axy = - (c_x[i]*c_y[i]          )*sin((x+0.5)/L)*sin((y+0.5)/L)*cos((z+0.5)/L);
		Ayx = + (c_x[i]*c_y[i]          )*sin((x+0.5)/L)*sin((y+0.5)/L)*cos((z+0.5)/L);
		Axz = - (c_x[i]*c_z[i]          )*sin((x+0.5)/L)*cos((y+0.5)/L)*sin((z+0.5)/L);
        Ayz = + (c_y[i]*c_z[i]          )*cos((x+0.5)/L)*sin((y+0.5)/L)*sin((z+0.5)/L);
        
		f_neq = -(w[i]*rho[index]*TAU*3.0)*(U_MAX/L)*(Axx + Ayy + Axy + Ayx + Axz + Ayz);
        feq = gpu_f_eq(w[i] * rho[index],
            3 * (ux[index] * c_x[i] + uy[index] * c_y[i] + uz[index] * c_z[i]),
            1 - (ux[index] * ux[index] + uy[index] * uy[index] + uz[index] * uz[index]));
        f[index_pop_d3q19(x, y, z, i)] = feq + f_neq;
        f_post[index_pop_d3q19(x, y, z, i)] = feq + f_neq;
    }
}


__host__
void bc_macr_collision_streaming(dfloat* f1, dfloat* f2, dfloat* rho, dfloat* ux, dfloat* uy, dfloat* uz,
    NodeTypeMap* ntm, bool save, int iter, cudaStream_t* stream)
{
    // blocks in grid
    dim3 grid(N_X / nThreads_X, N_Y / nThreads_Y, N_Z / nThreads_Z);
    // threads in block
    dim3 threads(nThreads_X, nThreads_Y, nThreads_Z);
    // size of shared memory (if needed)

    size_t shared_mem = threads.x*threads.y*threads.z*Q * sizeof(dfloat);
    gpu_bc_macr_collision_streaming << <grid, threads, 0, *stream>>> (f1, f2, rho, ux, uy, uz, ntm, save, iter);
    getLastCudaError("bc-macr-col-stream error");
}


__global__
//void __launch_bounds__(nThreads) 
void gpu_bc_macr_collision_streaming(dfloat * f1, dfloat * __restrict__ f2, dfloat * __restrict__ rho, dfloat * __restrict__ ux,
    dfloat * __restrict__ uy, dfloat* __restrict__ uz, NodeTypeMap* ntm, bool save, int iter)
{
    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= N_X || y >= N_Y || z >= N_Z)
        return;

    // apply boundary conditions
    if(ntm[index_scalar_d3(x, y, z)].get_BC_scheme() != BC_NULL)
        gpu_boundary_conditions(&ntm[index_scalar_d3(x, y, z)], f1, x, y, z);


    // adjacent coordinates
    const unsigned short int xp1 = (x + 1) % N_X;
    const unsigned short int yp1 = (y + 1) % N_Y;
    const unsigned short int zp1 = (z + 1) % N_Z;
    const unsigned short int xm1 = (N_X + x - 1) % N_X;
    const unsigned short int ym1 = (N_Y + y - 1) % N_Y;
    const unsigned short int zm1 = (N_Z + z - 1) % N_Z;

    // load populations
    dfloat f_aux[Q];
    for (char i = 0; i < Q; i++)
        f_aux[i] = f1[index_pop_d3q19(x, y, z, i)];
    
    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i]) / rho
    // uy = sum(f[i]*cy[i]) / rho
    // uz = sum(f[i]*cz[i]) / rho
    const dfloat rho_var = f_aux[0] + f_aux[1] + f_aux[2] + f_aux[3] + f_aux[4] + f_aux[5] + f_aux[6]
        + f_aux[7] + f_aux[8] + f_aux[9] + f_aux[10] + f_aux[11] + f_aux[12] + f_aux[13] + f_aux[14]
        + f_aux[15] + f_aux[16] + f_aux[17] + f_aux[18];
    const dfloat u_x_var = ((f_aux[1] + f_aux[7] + f_aux[9] + f_aux[13] + f_aux[15])
        - (f_aux[2] + f_aux[8] + f_aux[10] + f_aux[14] + f_aux[16])) / rho_var;
    const dfloat u_y_var = ((f_aux[3] + f_aux[7] + f_aux[11] + f_aux[14] + f_aux[17])
        - (f_aux[4] + f_aux[8] + f_aux[12] + f_aux[13] + f_aux[18])) / rho_var;
    const dfloat u_z_var = ((f_aux[5] + f_aux[9] + f_aux[11] + f_aux[16] + f_aux[18])
        - (f_aux[6] + f_aux[10] + f_aux[12] + f_aux[15] + f_aux[17])) / rho_var;

    if (save)
    {
        rho[index_scalar_d3(x, y, z)] = rho_var;
        ux[index_scalar_d3(x, y, z)] = u_x_var;
        uy[index_scalar_d3(x, y, z)] = u_y_var;
        uz[index_scalar_d3(x, y, z)] = u_z_var;
    }

    // calc for temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (u_x_var * u_x_var + u_y_var * u_y_var + u_z_var * u_z_var);
    const dfloat rho_w1 = rho_var * W_1;
    const dfloat rho_w2 = rho_var * W_2;
    const dfloat ux3 = 3 * u_x_var;
    const dfloat uy3 = 3 * u_y_var;
    const dfloat uz3 = 3 * u_z_var;

    // collision to f_post (f_aux)
    // f_aux = (1 - 1 / TAU) * f1 + (1 / TAU) * f_eq ->
    // f_aux = (1 - OMEGA) * f1 + OMEGA * f_eq ->
    // f_aux = T_OMEGA * f1 + OMEGA * f_eq
    f_aux[0] = T_OMEGA * f_aux[0] + OMEGA * gpu_f_eq(rho_var * W_0, 0, p1_muu15);
    f_aux[1] = T_OMEGA * f_aux[1] + OMEGA * gpu_f_eq(rho_w1, ux3, p1_muu15);
    f_aux[2] = T_OMEGA * f_aux[2] + OMEGA * gpu_f_eq(rho_w1, -ux3, p1_muu15);
    f_aux[3] = T_OMEGA * f_aux[3] + OMEGA * gpu_f_eq(rho_w1, uy3, p1_muu15);
    f_aux[4] = T_OMEGA * f_aux[4] + OMEGA * gpu_f_eq(rho_w1, -uy3, p1_muu15);
    f_aux[5] = T_OMEGA * f_aux[5] + OMEGA * gpu_f_eq(rho_w1, uz3, p1_muu15);
    f_aux[6] = T_OMEGA * f_aux[6] + OMEGA * gpu_f_eq(rho_w1, -uz3, p1_muu15);
    f_aux[7] = T_OMEGA * f_aux[7] + OMEGA * gpu_f_eq(rho_w2, ux3 + uy3, p1_muu15);
    f_aux[8] = T_OMEGA * f_aux[8] + OMEGA * gpu_f_eq(rho_w2, -ux3 - uy3, p1_muu15);
    f_aux[9] = T_OMEGA * f_aux[9] + OMEGA * gpu_f_eq(rho_w2, ux3 + uz3, p1_muu15);
    f_aux[10] = T_OMEGA * f_aux[10] + OMEGA * gpu_f_eq(rho_w2, -ux3 - uz3, p1_muu15);
    f_aux[11] = T_OMEGA * f_aux[11] + OMEGA * gpu_f_eq(rho_w2, uy3 + uz3, p1_muu15);
    f_aux[12] = T_OMEGA * f_aux[12] + OMEGA * gpu_f_eq(rho_w2, -uy3 - uz3, p1_muu15);
    f_aux[13] = T_OMEGA * f_aux[13] + OMEGA * gpu_f_eq(rho_w2, ux3 - uy3, p1_muu15);
    f_aux[14] = T_OMEGA * f_aux[14] + OMEGA * gpu_f_eq(rho_w2, -ux3 + uy3, p1_muu15);
    f_aux[15] = T_OMEGA * f_aux[15] + OMEGA * gpu_f_eq(rho_w2, ux3 - uz3, p1_muu15);
    f_aux[16] = T_OMEGA * f_aux[16] + OMEGA * gpu_f_eq(rho_w2, -ux3 + uz3, p1_muu15);
    f_aux[17] = T_OMEGA * f_aux[17] + OMEGA * gpu_f_eq(rho_w2, uy3 - uz3, p1_muu15);
    f_aux[18] = T_OMEGA * f_aux[18] + OMEGA * gpu_f_eq(rho_w2, -uy3 + uz3, p1_muu15);

    // streaming to f2
    // f2(x+cx, y+cy, z+cz, i) = f_post(x, y, z, i) 
    // the populations that shoudn't be streamed will be changed by the boundary conditions

    // streaming in directions orthogonal to X (cx=0)
    f2[index_pop_d3q19(x, y, z, 0)] = f_aux[0];
    f2[index_pop_d3q19(x, yp1, z, 3)] = f_aux[3];
    f2[index_pop_d3q19(x, ym1, z, 4)] = f_aux[4];
    f2[index_pop_d3q19(x, y, zp1, 5)] = f_aux[5];
    f2[index_pop_d3q19(x, y, zm1, 6)] = f_aux[6];
    f2[index_pop_d3q19(x, yp1, zp1, 11)] = f_aux[11];
    f2[index_pop_d3q19(x, ym1, zm1, 12)] = f_aux[12];
    f2[index_pop_d3q19(x, yp1, zm1, 17)] = f_aux[17];
    f2[index_pop_d3q19(x, ym1, zp1, 18)] = f_aux[18];
    // streaming forward in X direction (cx=1)
    f2[index_pop_d3q19(xp1, y, z, 1)] = f_aux[1];
    f2[index_pop_d3q19(xp1, yp1, z, 7)] = f_aux[7];
    f2[index_pop_d3q19(xp1, y, zp1, 9)] = f_aux[9];
    f2[index_pop_d3q19(xp1, ym1, z, 13)] = f_aux[13];
    f2[index_pop_d3q19(xp1, y, zm1, 15)] = f_aux[15];
    // streaming backwards in X direction (cx=-1)
    f2[index_pop_d3q19(xm1, y, z, 2)] = f_aux[2];
    f2[index_pop_d3q19(xm1, ym1, z, 8)] = f_aux[8];
    f2[index_pop_d3q19(xm1, y, zm1, 10)] = f_aux[10];
    f2[index_pop_d3q19(xm1, yp1, z, 14)] = f_aux[14];
    f2[index_pop_d3q19(xm1, y, zp1, 16)] = f_aux[16];
    
}


__host__
void update_rho_u(dfloat* f, dfloat* rho, dfloat* u_x, dfloat* u_y, dfloat* u_z)
{
    // blocks in grid
    dim3 grid(N_X / nThreads_X, N_Y / nThreads_Y, N_Z / nThreads_Z);
    // threads in block
    dim3 threads(nThreads_X, nThreads_Y, nThreads_Z);

    gpu_update_rho_u << <grid, threads >> > (f, rho, u_x, u_y, u_z);
    getLastCudaError("macroscopics error");
}


__global__
void gpu_update_rho_u(dfloat* f, dfloat* rho, dfloat* __restrict__ u_x, dfloat* __restrict__ u_y, dfloat* __restrict__ u_z)
{
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= N_X || y >= N_Y || z >= N_Z)
        return;

    // load populations
    dfloat f_aux[Q];
    for (unsigned char i = 0; i < Q; i++)
        f_aux[i] = f[index_pop_d3q19(x, y, z, i)];

    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i]) / rho
    // uy = sum(f[i]*cy[i]) / rho
    // uz = sum(f[i]*cz[i]) / rho
    const dfloat rho_var = f_aux[0] + f_aux[1] + f_aux[2] + f_aux[3] + f_aux[4] + f_aux[5] + f_aux[6]
        + f_aux[7] + f_aux[8] + f_aux[9] + f_aux[10] + f_aux[11] + f_aux[12] + f_aux[13] + f_aux[14]
        + f_aux[15] + f_aux[16] + f_aux[17] + f_aux[18];
    const dfloat u_x_var = ((f_aux[1] + f_aux[7] + f_aux[9] + f_aux[13] + f_aux[15])
        - (f_aux[2] + f_aux[8] + f_aux[10] + f_aux[14] + f_aux[16])) / rho_var;
    const dfloat u_y_var = ((f_aux[3] + f_aux[7] + f_aux[11] + f_aux[14] + f_aux[17])
        - (f_aux[4] + f_aux[8] + f_aux[12] + f_aux[13] + f_aux[18])) / rho_var;
    const dfloat u_z_var = ((f_aux[5] + f_aux[9] + f_aux[11] + f_aux[16] + f_aux[18])
        - (f_aux[6] + f_aux[10] + f_aux[12] + f_aux[15] + f_aux[17])) / rho_var;

    rho[index_scalar_d3(x, y, z)] = rho_var;
    u_x[index_scalar_d3(x, y, z)] = u_x_var;
    u_y[index_scalar_d3(x, y, z)] = u_y_var;
    u_z[index_scalar_d3(x, y, z)] = u_z_var;
}


__host__
dfloat residual(dfloat * u_x, dfloat * u_y, dfloat * u_z, dfloat * u_x_res, dfloat * u_y_res, dfloat * u_z_res)
{
    dfloat den = 0.0, num = 0.0;

    for (unsigned int z = 0; z < N_Z; z++)
        for (unsigned int y = 0; y < N_Y; y++)
            for (unsigned int x = 0; x < N_X; x++)
            {
                const dfloat diff_ux = u_x[index_scalar_d3(x, y, z)] - u_x_res[index_scalar_d3(x, y, z)];
                const dfloat diff_uy = u_y[index_scalar_d3(x, y, z)] - u_y_res[index_scalar_d3(x, y, z)];
                const dfloat diff_uz = u_z[index_scalar_d3(x, y, z)] - u_z_res[index_scalar_d3(x, y, z)];

                num += std::sqrt(diff_ux * diff_ux + diff_uy * diff_uy + diff_uz * diff_uz);
                den += std::sqrt(u_x[index_scalar_d3(x, y, z)] * u_x[index_scalar_d3(x, y, z)]
                    + u_y[index_scalar_d3(x, y, z)] * u_y[index_scalar_d3(x, y, z)]
                    + u_z[index_scalar_d3(x, y, z)] * u_z[index_scalar_d3(x, y, z)]);
            }
    if (den != 0)
        return (num / den);
    else
        return 1.0;
}
dfloat residual_parallel(dfloat* u_x, dfloat* u_y, dfloat* u_z, dfloat* u_x_res, dfloat* u_y_res, dfloat* u_z_res, dfloat* num_gpu, dfloat* den_gpu)
{
    // blocks in grid
    dim3 grid(N_X / SIZE_X_RES, N_Y / SIZE_Y_RES, N_Z / SIZE_Z_RES);
    // threads in block
    dim3 threads(SIZE_X_RES, SIZE_Y_RES, SIZE_Z_RES);

    // alloc memory
    dfloat res, num_v = 0, den_v = 0;
    dfloat* num, *den;
    checkCudaErrors(cudaMallocHost((void**)&den, mem_size_res));
    checkCudaErrors(cudaMallocHost((void**)&num, mem_size_res));

    gpu_residual << <grid, threads >> > (u_x, u_y, u_z, u_x_res, u_y_res, u_z_res, num_gpu, den_gpu);
    checkCudaErrors(cudaMemcpy(num, num_gpu, mem_size_res, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(den, den_gpu, mem_size_res, cudaMemcpyDeviceToHost));

    for (int k = 0; k < SIZE_Z_RES; k++)
        for (int j = 0; j < SIZE_Y_RES; j++)
            for (int i = 0; i < SIZE_X_RES; i++)
            {
                size_t index = SIZE_X_RES * (SIZE_Y_RES*k + j) + i;
                num_v += num[index];
                den_v += den[index];
            }
    if (den_v != 0)
        return num_v / den_v;
    return 1.0;
}


__global__
void gpu_residual(dfloat * u_x, dfloat * u_y, dfloat * u_z, dfloat * u_x_res, dfloat * u_y_res, dfloat * u_z_res, dfloat* num, dfloat* den)
{
    dfloat num_v = 0, den_v = 0;

    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;

    for (unsigned short int k = z; k < z + SIZE_Z_RES; k++)
    {
        for (unsigned short int j = y; j < y + SIZE_Y_RES; j++)
        {
            for (unsigned short int i = 0; i < x + SIZE_X_RES; i++)
            {
                const dfloat diff_ux = u_x[index_scalar_d3(i, j, k)] - u_x_res[index_scalar_d3(i, j, k)];
                u_x_res[index_scalar_d3(i, j, k)] = u_x[index_scalar_d3(i, j, k)];
                const dfloat diff_uy = u_y[index_scalar_d3(i, j, k)] - u_y_res[index_scalar_d3(i, j, k)];
                u_y_res[index_scalar_d3(i, j, k)] = u_y[index_scalar_d3(i, j, k)];
                const dfloat diff_uz = u_z[index_scalar_d3(i, j, k)] - u_z_res[index_scalar_d3(i, j, k)];
                u_z_res[index_scalar_d3(i, j, k)] = u_z[index_scalar_d3(i, j, k)];

                num_v += std::sqrt(diff_ux * diff_ux + diff_uy * diff_uy + diff_uz * diff_uz);
                den_v += std::sqrt(u_x[index_scalar_d3(i, j, k)] * u_x[index_scalar_d3(i, j, k)]
                    + u_y[index_scalar_d3(i, j, k)] * u_y[index_scalar_d3(i, j, k)]
                    + u_z[index_scalar_d3(i, j, k)] * u_z[index_scalar_d3(i, j, k)]);
            }
        }
    }

    num[index_residual_d3(x, y, z)] = num_v;
    den[index_residual_d3(x, y, z)] = den_v;
}


__host__
void equalize_vel(dfloat* u_x, dfloat* u_y, dfloat* u_z, dfloat* u_x_0, dfloat* u_y_0, dfloat* u_z_0)
{
    for (unsigned int z = 0; z < N_Z; z++)
        for (unsigned int y = 0; y < N_Y; y++)
            for (unsigned int x = 0; x < N_X; x++)
            {
                u_x_0[index_scalar_d3(x, y, z)] = u_x[index_scalar_d3(x, y, z)];
                u_y_0[index_scalar_d3(x, y, z)] = u_y[index_scalar_d3(x, y, z)];
                u_z_0[index_scalar_d3(x, y, z)] = u_z[index_scalar_d3(x, y, z)];
            }
}