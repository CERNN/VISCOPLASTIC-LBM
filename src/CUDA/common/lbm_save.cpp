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

#include "lbm_save.h"

std::string get_var_filename(const std::string id, const std::string var_name, unsigned int n_steps)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
    if (n_steps != 0)
        for (n_zeros = 0; n_steps * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 6;

    // generates the file name as "PATH/id_var_name000000.bin"
    std::string str_file = PATH_DATA + id + "_" + var_name;
    for (unsigned int i = 0; i < n_zeros; i++)
        str_file += "0";
    str_file += std::to_string(n_steps) + ".bin";
    
    return str_file;
}


void save_variable_bin(const std::string id, const std::string var_name, unsigned int n_steps, dfloat * var, size_t mem_size)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
     
    if (n_steps != 0)
        for (n_zeros = 0; n_steps * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 6;

    // generates the file name as "PATH/id_var_name000000.bin"
    std::string str_file = PATH_DATA + id + "_" + var_name;
   
    for (unsigned int i = 0; i < n_zeros; i++)
        str_file += "0";
    str_file += std::to_string(n_steps) + ".bin";
   
    FILE* outFile = nullptr;

    outFile = fopen(str_file.c_str(), "wb");
    if(outFile != nullptr)
        fwrite(var, mem_size, 1, outFile);
    else
    {
        printf("Error saving %s (probably wrong path)!", str_file.c_str());
    }
    fclose(outFile);
}


#ifdef D3Q19

void save_sim_inf_d3(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device)
{
    std::string str_inf = PATH_DATA + id + "_inf.txt"; // generate file name (with path)

    FILE* outFile = nullptr;

    outFile = fopen(str_inf.c_str(), "w");

    if (outFile == nullptr)
    {
        printf("Error saving simulation info, invalid output path\n");
        return;
    }

    fprintf(outFile, "SIMULATION INFORMATION:\n");
    fprintf(outFile, "                 NX: %d\n", N_X);
    fprintf(outFile, "                 NY: %d\n", N_Y);
    fprintf(outFile, "                 NZ: %d\n", N_Z);
    fprintf(outFile, "           Reynolds: %.2f\n", REYNOLDS);
    fprintf(outFile, "                Tau: %.6e\n", TAU);
    fprintf(outFile, "               Umax: %.6e\n", U_MAX);
    fprintf(outFile, "           Residual: %.6e\n", res);    
    fprintf(outFile, "     Residual steps: %d\n", (RESID? N_RESID : 0));
    fprintf(outFile, "         Save steps: %d\n", N_SAVE);
    fprintf(outFile, "             Nsteps: %d\n", n_steps);
    fprintf(outFile, "              MLUPS: %.1f\n", mlups);
    fprintf(outFile, "          Bandwidht: %.1f (Gb/s)\n\n", bandwidth);

    fprintf(outFile, "CUDA INFORMATION\n");
    fprintf(outFile, "               name: %s\n", device.name);
    fprintf(outFile, "    multiprocessors: %d\n", device.multiProcessorCount);
    fprintf(outFile, " compute capability: %d.%d\n", device.major, device.minor);
    fprintf(outFile, "        ECC enabled: %d\n", device.ECCEnabled);
    fprintf(outFile, "            threads: (%d, %d, %d)", nThreads_X, nThreads_Y, nThreads_Z);
    fclose(outFile);
}
#else

void save_sim_inf_d3(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device)
{
    printf("Not using D3Q19! Unable to execute save_sim_inf_d3()\n");
}

#endif // D3Q19



#ifdef D2Q9

void save_sim_inf(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device)
{
    std::string str_inf = PATH_DATA + id + "_inf.txt"; // generate file name (with path)

    FILE* outFile = nullptr;

    outFile = fopen(str_inf.c_str(), "w");

    if (outFile == nullptr)
    {
        printf("Error saving simulation info, invalid output path\n");
        return;
    }

    fprintf(outFile, "SIMULATION INFORMATION:\n");
    fprintf(outFile, "                 NX: %d\n", N_X);
    fprintf(outFile, "                 NY: %d\n", N_Y);
    fprintf(outFile, "           Reynolds: %.2f\n", REYNOLDS);
    fprintf(outFile, "                Tau: %.6e\n", TAU);
    fprintf(outFile, "               Umax: %.6e\n", U_MAX);
    fprintf(outFile, "           Residual: %.6e\n", res);    
    fprintf(outFile, "     Residual steps: %d\n", (RESID? N_RESID : 0));
    fprintf(outFile, "         Save steps: %d\n", N_SAVE);
    fprintf(outFile, "             Nsteps: %d\n", n_steps);
    fprintf(outFile, "              MLUPS: %.1f\n", mlups);
    fprintf(outFile, "          Bandwidht: %.1f (Gb/s)\n\n", bandwidth);

    fprintf(outFile, "CUDA INFORMATION\n");
    fprintf(outFile, "               name: %s\n", device.name);
    fprintf(outFile, "    multiprocessors: %d\n", device.multiProcessorCount);
    fprintf(outFile, " compute capability: %d.%d\n", device.major, device.minor);
    fprintf(outFile, "        ECC enabled: %d\n", device.ECCEnabled);
    fprintf(outFile, "            threads: (%d, %d)", nThreads_X, nThreads_Y);
    fclose(outFile);
}

void save_ux_uy(const std::string id, dfloat* ux, dfloat* uy)
{
    std::string str_ux = PATH_DATA + id + "_ux_c" + EXT;    // generate u_x file name (with path)
    std::string str_uy = PATH_DATA + id + "_uy_c" + EXT;    // generate u_y file name (with path)

    std::fstream outFile_ux(str_ux.c_str(), std::fstream::out);
    std::fstream outFile_uy(str_uy.c_str(), std::fstream::out);

    int x, y;
    
    for (x = 0; x < N_X; x++)
    {
        // if the number of nodes is even, the value saved is de average of the ones in indexes [y][N_Y/2] and [x][N_Y/2-1]
        dfloat u_y_i;
        if (N_Y % 2)
            u_y_i = uy[index_scalar(x, N_Y / 2)];
        else
            u_y_i = (uy[index_scalar(x, N_Y / 2)] + uy[index_scalar(x, N_Y / 2 - 1)]) / 2;

        // fix precision to 6 houses
        outFile_uy << std::fixed;
        outFile_uy << ((dfloat)x / (N_X - 1)) << SEP;
        // fix scientific notation
        outFile_uy << std::scientific;
        outFile_uy << u_y_i / U_MAX << std::endl;       // writes normalized velocity
    }

    for (y = 0; y < N_Y; y++)
    {
        // if the number of nodes is even, the value saved is de average of the ones in indexes [N_X/2][y] and [N_X/2-1][y]
        dfloat u_x_i;
        if (N_X % 2)
            u_x_i = ux[index_scalar(N_X / 2, y)];
        else
            u_x_i = (ux[index_scalar(N_X / 2, y)] + ux[index_scalar(N_X / 2 - 1, y)]) / 2;

        outFile_ux << std::fixed;
        outFile_ux << ((dfloat)y / (N_Y - 1)) << SEP;
        outFile_ux << std::scientific;
        outFile_ux << u_x_i / U_MAX<< std::endl;    // writes normalized velocity

    }
    
    outFile_ux.close();
    outFile_uy.close();
    
}


void save_ux(const std::string id, dfloat* ux)
{
    std::string str_ux = PATH_DATA + id + "_ux_c" + EXT;    // generate u_x file name (with path)

    std::fstream outFile_ux(str_ux.c_str(), std::fstream::out);
    
    for (int y = 0; y < N_Y; y++)
    {
        // if the number of nodes is even, the value saved is de average of the ones indexes [N_X/2][y] and [N_X/2-1][y]
        dfloat u_x_i;
        if (N_X % 2)
            u_x_i = ux[index_scalar(N_X / 2, y)];
        else
            u_x_i = (ux[index_scalar(N_X / 2, y)] + ux[index_scalar(N_X / 2 - 1, y)]) / 2;

        outFile_ux << std::fixed;
        outFile_ux << ((dfloat)y / (N_Y - 1)) << SEP;
        outFile_ux << std::scientific;
        outFile_ux << u_x_i / U_MAX << std::endl;   // writes normalized velocity

    }

    outFile_ux.close();
}

#else
void save_sim_inf(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device)
{
    printf("Not using D2Q9! Unable to execute save_sim_inf()\n");
}

void save_ux_uy(const std::string id, dfloat* ux, dfloat* uy)
{
    printf("Not using D2Q9! Unable to execute save_ux_uy()\n");
}

void save_ux(const std::string id, dfloat* ux)
{
    printf("Not using D2Q9! Unable to execute save_ux()\n");
}

#endif // D2Q9