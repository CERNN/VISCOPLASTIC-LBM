#ifndef LBM_SAVE_H
#define LBM_SAVE_H

#include <fstream>      // for saving
#include <string>
#include "func_idx.cuh"

/*
*   Get variable filename
*   \param id: simulation's id
*   \param var_name: name of the variable
*   \param n_steps: number of steps of the file
*/
std::string get_var_filename(const std::string id, const std::string var_name, unsigned int n_steps);



/*
*   Save variables values in binary format
*   \param id: simulation's id
*   \param var_name: name of the variable
*   \param n_steps: number of steps of the simulation
*   \param var: array values to save
*   \param mem_size: number of bytes to save
*   Obs1.: check PC endianess (currently little endian)
*   Obs2.: the initial position of the array is x=0 and y=0 and z=0, so the variables starts on SWF and ends in NEB
*/
void save_variable_bin(const std::string id, const std::string var_name, unsigned int n_steps, dfloat* var, size_t mem_size);


/*
*   Save simulation info
*   \param id: simulation's id
*   \param mlups: million lattice updates per second
*   \param res: simulation's residual
*   \param n_steps: simulation's times steps
*   \param device: CUDA device used for simulation
*/
void save_sim_inf(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device);


/*
*   Save simulation info
*   \param id: simulation's id
*   \param mlups: million lattice updates per second
*   \param res: simulation's residual
*   \param n_steps: simulation's times steps
*   \param device: CUDA device used for simulation
*/
void save_sim_inf_d3(const std::string id, const dfloat mlups, const dfloat bandwidth, const dfloat res, const int n_steps, cudaDeviceProp device);


/*
*   Saves normalized u_x on x = 0.5 and u_y on y = 0.5 values of the program (for cavity 2d)
*   \param id: simulation's id
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values to be saved
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be saved
*/
void save_ux_uy(const std::string id, dfloat* ux_gpu, dfloat* uy_gpu);


/*
*   Saves normalized u_x on x = 0.5 values of the program (for parallel plates)
*   \param id: simulation's id
*   \param u_y[(N_X, N_Y)]: nodes' x velocity values to be saved
*/
void save_ux(const std::string id, dfloat* ux_gpu);


#endif // LBM_SAVE_H
