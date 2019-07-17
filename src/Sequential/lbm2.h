#ifndef LBM2_H
#define LBM2_H

/*
    THESE INCLUDES ARE IN "pch.h" FILE

#include <fstream>      // for saving
#include <iostream>
#include <iomanip>      // for fixed houses on saving
#include <cmath>        // for sqrt on residual
*/

#define PATH_DATA "./tests/"         // path to save simulation's data
#define EXT ".csv"                      // file to save extension
#define SEP "\t"                        // csv separator
#define ID_PROG "02"                    // program's id


/// defines for LBM related variables
constexpr unsigned int N = (256);      // size of the grid
constexpr unsigned int N_X = N;         // size x of the grid
constexpr unsigned int N_Y = N;         // size y of the grid
constexpr unsigned int Q = 9;           // number of velocities

constexpr double C_S_INV_SQR = 3.0;     // (sound velocity) ^ -2
constexpr double U_W_TOP = 0.05;        // top wall's velocity

constexpr double REYNOLDS = 100.0;       // Reynolds number
constexpr double TAU = 0.5 + 3 *        // Relaxation time calculated for
    (((double)N) * U_W_TOP / REYNOLDS); // the Reynolds number
//constexpr double TAU = 0.9;
constexpr double OMEGA = 1.0 / TAU;     // (tau) ^ -1


constexpr double RHO_0 = 1;                     // rho 0 (reference)
constexpr double RHO_OUT = RHO_0;               // out fluid's density for parallel plates
constexpr double RHO_IN = RHO_OUT + RHO_0 * 12  // in fluid's density for parallel plates
* U_W_TOP * (TAU - 0.5) / (N);

constexpr double RESID_MAX = 1e-4;      // simulation maximal residual


/*
--------------------------

------ POPULATIONS -------

------  6   2   5   ------

------  3   0   1   ------

------  7   4   8   ------

--------------------------
*/


constexpr double W_0 = 4.0 / 9.0;       // population 0 weight
constexpr double W_1 = 1.0 / 9.0;       // adjacent populations (1, 2, 3, 4) weight
constexpr double W_2 = 1.0 / 36.0;      // diagonal populations (5, 6, 7, 8) weight

// velocities weight vector
constexpr double w[Q] = {W_0, W_1, W_1, W_1, W_1, W_2, W_2, W_2, W_2};

// populations velocities vector, excluding population zero
constexpr char c_x[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr char c_y[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};



/*
*   Evaluate the position of the element of a 2D matrix ([N_X][N_Y]) in a 1D array
*   \param x: value in first dimension
*   \param y: value in second dimension
*   \return element index
*/
size_t inline scalar_index2d(unsigned int x, unsigned int y)
{
    return N_Y * x + y;
}


/*
*   Evaluate the position of the population of a 3D matrix ([N_X][N_Y][Q-1]) in a 1D array 
*   \param x: value in first dimension
*   \param y: value in second dimension
*   \param d: population number
*   \return element index
*/

size_t inline scalar_index_pop(unsigned int x, unsigned int y, unsigned int d)
{
    return (Q - 1) * (N_Y * x + y) + (d - 1);
}


/*
*   Evaluate the population of equilibrium
*   \param rho: density
*   \param u_c: scalar product of velocity and the discretized velocity i
*   \param u_u: scalar product of velocity and itself
*   \param w_i: population's weight
*   \return equilibrium population
*/
const double inline f_eq(const double rho, const double u_c, const double u_u, const double w_i)
{
    return (rho * w_i * (1.0 + u_c * 3 * (1.0 + u_c * 3 / 2.0) -
        u_u * 3 / 2.0));
}


/*
*   Evaluate the population of equilibrium
*   \param rho: density
*   \param u_x: x velocity
*   \param u_y: y velocity
*   \param i: population's number
*   \return equilibrium population
*/
const double inline f_eq_generic(const double rho, const double u_x, const double u_y, const int i)
{
    double u_c = u_x * c_x[i] + u_y * c_y[i];
    double u_u = u_x * u_x + u_y * u_y;
    double w_i = w[i];
    return (rho * w_i * (1.0 + u_c * 3 * (1.0 + u_c * 3 / 2.0) -
        u_u * 3 / 2.0));
}


/*
*   Applies boundary conditions for north wall
*   \param f: node's populations 1-8
*   \param f0: node's population 0
*   \param u_x: x velocity
*   \param u_y: y velocity
*/
void inline boundary_conditions_N(double* f, const double f0, const double u_x, const double u_y)
{
    const double rho = 1 / (1 + u_y) * (f0 + f[0] + f[2] + 2 * (f[1] + f[4] + f[5]));

    f[3] = f[1] - 2.0 / 3 * rho * u_y;
    f[6] = f[4] + 1.0 / 2 * (f[0] - f[2]) - 1.0 / 6 * rho * u_y - 1.0 / 2 * rho * u_x;
    f[7] = f[5] - 1.0 / 2 * (f[0] - f[2]) - 1.0 / 6 * rho * u_y + 1.0 / 2 * rho * u_x;
}


/*
*   Applies boundary conditions for east wall
*   \param f: node's populations 1-8
*   \param f0: node's population 0
*   \param u_x: x velocity
*   \param u_y: y velocity
*/
void inline boundary_conditions_E(double* f, const double f0, const double u_x, const double u_y)
{
    const double rho = 1 / (1 + u_x) * (f0 + f[1] + f[3] + 2 * (f[0] + f[4] + f[7]));

    f[2] = f[0] - 2.0 / 3 * rho * u_x;
    f[5] = f[7] - 1.0 / 2 * (f[1] - f[3]) - 1.0 / 6 * rho * u_x + 1.0 / 2 * rho * u_y;
    f[6] = f[4] + 1.0 / 2 * (f[1] - f[3]) - 1.0 / 6 * rho * u_x - 1.0 / 2 * rho * u_y;
}


/*
*   Applies boundary conditions for south wall
*   \param f: node's populations 1-8
*   \param f0: node's population 0
*   \param u_x: x velocity
*   \param u_y: y velocity
*/
void inline boundary_conditions_S(double* f, const double f0, const double u_x, const double u_y)
{
    const double rho = 1 / (1 + u_y) * (f0 + f[0] + f[2] + 2 * (f[3] + f[6] + f[7]));

    f[1] = f[3] + 2.0 / 3 * rho * u_y;
    f[4] = f[6] - 1.0 / 2 * (f[0] - f[2]) + 1.0 / 6 * rho * u_y + 1.0 / 2 * rho * u_x;
    f[5] = f[7] + 1.0 / 2 * (f[0] - f[2]) + 1.0 / 6 * rho * u_y - 1.0 / 2 * rho * u_x;
}


/*
*   Applies boundary conditions for west wall
*   \param f: node's populations 1-8
*   \param f0: node's population 0
*   \param u_x: x velocity
*   \param u_y: y velocity
*/
void inline boundary_conditions_W(double* f, const double f0, const double u_x, const double u_y)
{
    const double rho = 1 / (1 - u_x) * (f0 + f[1] + f[3] + 2 * (f[2] + f[5] + f[6]));

    f[0] = f[2] + 2.0 / 3 * rho * u_x;
    f[4] = f[6] - 1.0 / 2 * (f[1] - f[3]) + 1.0 / 6 * rho * u_x + 1.0 / 2 * rho * u_y;
    f[7] = f[5] + 1.0 / 2 * (f[1] - f[3]) + 1.0 / 6 * rho * u_x - 1.0 / 2 * rho * u_y;
}


/*
*   Initializes populations
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to be initialized
*   \param f_post[(N_X, N_Y, (Q-1))]: post populations from 1 to 8 to be initialized
*   \param f_0[(N_X, N_Y)]: populations 0 to be initialized
*   \param rho[(N_X, N_Y)]: nodes' density to initialize
*   \param u_x[(N_X, N_Y)]: nodes' x velocity to initialize
*   \param u_y[(N_X, N_Y)]: nodes' y velocity to initialize
*/
void initialisation(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Performs collision
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to perform collision
*   \param f_post[(N_X, N_Y, (Q-1))]: post populations from 1 to 8
*   \param f_0[(N_X, N_Y)]: populations 0 to perform collision
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*/
void collision(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Performs collision
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to perform collision
*   \param f_post[(N_X, N_Y, (Q-1))]: post populations from 1 to 8
*   \param f_0[(N_X, N_Y)]: populations 0 to perform collision
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*/
void collision_generic(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Performs collision with regularized pre-collision distribution functions
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to perform collision
*   \param f_post[(N_X, N_Y, (Q-1))]: post populations from 1 to 8
*   \param f_0[(N_X, N_Y)]: populations 0 to perform collision
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*/
void collision_regularized(double* f, double* f_post, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Performs streaming
*   \param f1[(N_X, N_Y, (Q-1))]: matrix of populations to stream to
*   \param f2[(N_X, N_Y, (Q-1))]: matrix of populations to stream from
*/
void streaming(double* f1, double* f2);


/*
*   Performs streaming
*   \param f1[(N_X, N_Y, (Q-1))]: matrix of populations to stream to
*   \param f2[(N_X, N_Y, (Q-1))]: matrix of populations to stream from
*/
void streaming_generic(double* f1, double* f2);


/*
*   Updates macroscopics and then performs collision and streaming
*   \param f1[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to perform collision and stream for
*   \param f2[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to perform collision and stream to
*   \param f_0[(N_X, N_Y)]: populations 0 to perform collision
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*/
void macr_collision_streaming(double* f1, double* f2, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Applies NEBB boundary conditions for parallel plates 
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to apply boundary conditions
*   \param f_0[(N_X, N_Y)]: populations 0
*   \param f_post[(N_X, N_Y, (Q-1))]: post populations from 1 to 8
*   \param rho[(N_X, N_Y)]: nodes' density values
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values
*/
void boundary_conditions_nebb_pp2d(double* f, double* f0, double* f_post, double* rho, double* u_x, double* u_y);


/*
*   Applies NEBB boundary conditions for cavity 2d
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8 to apply boundary conditions
*   \param f_0[(N_X, N_Y)]: populations 0 to apply boundary conditions
*/
void boundary_conditions_nebb_c2d(double* f, double* f_0);


/*
*   Update densisty and velocity of all nodes
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8
*   \param f_0[(N_X, N_Y)]: populations 0
*   \param rho[(N_X, N_Y)]: nodes' density values to be updated
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values to be updated
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be updated
*/
void update_rho_u(double* f, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Update densisty and velocity of all nodes
*   \param f[(N_X, N_Y, (Q-1))]: populations from 1 to 8
*   \param f_0[(N_X, N_Y)]: populations 0
*   \param rho[(N_X, N_Y)]: nodes' density values to be updated
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values to be updated
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be updated
*/
void update_rho_u_generic(double* f, double* f_0, double* rho, double* u_x, double* u_y);


/*
*   Calculate the residual of two populations
*   \param u_y[(N_X, N_Y)]: nodes' current x velocity values
*   \param u_y[(N_X, N_Y)]: nodes' current y velocity values
*   \param u_x_0[(N_X, N_Y)]: nodes' reference x velocity values
*   \param u_y_0[(N_X, N_Y)]: nodes' reference y velocity values
*   \return residual value
*/
const double residual(double* u_x, double* u_y, double* u_x_res, double* u_y_res);


/*
*   Calculates average x velocity on x=0.5
*   \param u_x[(N_X, N_Y)]: nodes' current x velocity values
*   \return average x velocity on x=0.5
*/
const double u_med(double* u_x);


/*
*   Equalizes velocities (u_x_0 = u_x and u_y_0 = u_y)
*   \param u_y[(N_X, N_Y)]: nodes' x velocity values (reference)
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values (reference)
*   \param u_x_0[(N_X, N_Y)]: nodes' x velocity to equalize
*   \param u_y_0[(N_X, N_Y)]: nodes' y velocity to equalize
*/
void equalize_vel(double* u_x, double* u_y, double* u_x_0, double* u_y_0);


/*
*   Save the variables values of the program
*   \param id: simulation's id
*   \param n_steps: number of steps of the simulation
*   \param rho[(N_X, N_Y)]: nodes' density values to be saved
*   \param u_y[(N_X, N_Y)]: nodes' x velocity values to be saved
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be saved
*/
void save(const std::string id, const unsigned int n_steps, double* rho, double* u_x, double* u_y);


/*
*   Save parameters of the program
*   \param id: simulation's id
*   \param mlups: million lattice updates per second
*   \param res: simulations' residual 
*/
void save_inf(const std::string id, const double mlups, const double res, const int n_steps);


/*
*   Saves normalized u_x on x = 0.5 and u_y on y = 0.5 values of the program (for cavity 2d)
*   \param id: simulation's id
*   \param u_x[(N_X, N_Y)]: nodes' x velocity values to be saved
*   \param u_y[(N_X, N_Y)]: nodes' y velocity values to be saved
*/
void save_ux_uy(const std::string id, double* u_x, double* u_y);


/*
*   Saves u_x/u_med on x = 0.5 values of the program (for parallel plates)
*   \param id: simulation's id
*   \param u_y[(N_X, N_Y)]: nodes' x velocity values to be saved
*/
void save_ux(const std::string id, double* u_x);


#endif // LBM2_H
