# LBM Viscoplastic

This is a LBM (Lattice Boltzmann method) project for flow simulation, using C/C++ and CUDA API. The velocity sets supported are D3Q19 and D3Q27. 

If you are not familiarized with the method, we strongly recommend the book _The Lattice Boltzmann Method: Principles and Practice_ by Kruger et al. as a guide.

For software citation use _PERFORMANCE ANALYSIS OF THE LATTICE BOLTZMANN METHOD IMPLEMENTATION ON GPU_, by Oliveira et al., XL CILAMCE ([CILAMCE site](https://www.cilamce2019.com.br), [Research gate](https://www.researchgate.net/publication/341522565_PERFORMANCE_ANALYSIS_OF_THE_LATTICE_BOLTZMANN_METHOD_IMPLEMENTATION_ON_GPU))

## Features

The LBM features supported are:

* Boundary Conditions
    * Zou-He (velocity or pressure)
    * Bounce-back (velocity)
    * Free slip

* Collision
    * BGK operator
    * Regularization
    * Constant force

* Velocity Sets
    * D3Q19
    * D3Q27

Other features can be implemented or changed. The file [updating.md](./doc/updating.md) gives more details on that.

## Compilation

The requirements are:
* Nvidia drivers must be installed
* CUDA API must be installed

Both can be obtained in "CUDA Toolkit", provided by Nvidia.

The code supports Nvidia's GPUs with compute capability 3.5 or higher. The program runs in only one GPU, multi-GPU support is in development.

For compilation, a [bash file](./src/CUDA/compile.sh) is provided. It contains the commands used to compile and the instructions for altering it according to the GPU compute capability and the arguments to pass to it.

## Simulation

The output of the simulations are binary files with the content of macroscopics (density, velocity, etc.), an information file with the simulation parameters (lattice size, tau, velocity set, etc.), It is also provided an option to output treated data, values obtained by the treatment of simulation macroscopics, as residual and average density. The simulation information and the treated data are also printed on screen. To convert from binary to interpretable data, a Python application is provided. "Post Processing" gives more details on that.

More details on code usage is described in [updating.md](./doc/updating.md). The file describes how to change simulation parameters, boundary conditions, collision scheme, etc.

## Documentation

The documentation of the source files is made in markDown files ([files.md](./doc/files.md) and [updating.md](./doc/updating.md)) and in header files (_.h_), with each function presenting _@brief_, _@param_ and _@return_, describing what it does, its parameters and what it returns.

## Post Processing

Since the program exports macroscopics in binary format, it is necessary to process it. For that, Python source files are provided. _python3_ is required and the packages dependecies are:
* glob
* numpy
* os
* pyevtk
* matplotlib

Two example files are presented in _CUDA/Post Processing/_ folder. Basic data treatment, exportation and plot is implemented by these files. Graphics f(x)=y, heatmaps and exportation to VTK and .csv are the examples provided. Feel free to alter these to match the required data processing.

Implementations details and more informations can be obtained in the source files and in [files.md](./doc/files.md).

## License

This software is provided under the [GPLv2 license](./LICENSE.txt).

## Contact

For bug report or issue adressing, usage of git resources (issues/pull request) is encouraged. Contact via email: _waine@alunos.utfpr.edu.br_ and/or _cernn-ct@utfpr.edu.br_.
