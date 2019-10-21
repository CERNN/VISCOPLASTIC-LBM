# LBM

Below is described a brief of the content of the headers/source files in _CUDA/_ folder.

#### [LBM](../src/CUDA/lbm.h)
LBM collision, streaming and macroscopics update. Special treatment for non local boundary conditions.
#### [LBM Initialization](../src/CUDA/lbmInitialization.h)
LBM  initial field of macroscopics, initial populations, methods for loading macroscopics or populations from binary files.
#### [LBM Report](../src/CUDA/lbmReport.h)
Report simulation parameters and save macroscopics/populations.
#### [Boundary Conditions Builder](../src/CUDA/boundaryConditionsBuilder.h)
Declaration of the function to be implemented by the source files in the _CUDA/boundaryConditionsBuilders/_ folder.
#### [Boundary Conditions Handler](../src/CUDA/boundaryConditionsHandler.h)
Switch cases for the call of boundary conditions functions.
#### [Treat Data](../src/CUDA/treatData.h)
Treat data from simulation, as calculate residual, and report/save it.
#### [Var](../src/CUDA/var.h)
Simulation's options and variables.
#### [Global Functions](../src/CUDA/globalFunctions.cu)
Inline functions used by many functions, as index calculation, equilibrium population, etc.
#### [Main](../src/CUDA/main.cu)
Implementation of _main()_ function.

## Velocity sets

#### [D3Q19](../src/CUDA/velocitySets/D3Q19.h) and [D3Q27](../src/CUDA/velocitySets/D3Q27.h)
Velocity set values, as populations' directions and weights.

## Structs

#### [Macroscopics](../src/CUDA/structs/macroscopics.h)
Struct for macroscopics declaration and to allocate/free it.
#### [Populations](../src/CUDA/structs/populations.h)
Struct for declaration of populations and boundary conditions map and to allocate/free it.
#### [Macroscopics Processing](../src/CUDA/structs/macrProc.h)
Struct for declaration of simulation variables to be processed (as residual and average density) and to allocate/free it, if required.
#### [Node Type Map](../src/CUDA/structs/nodeTypeMap.h)
Struct with implemantation of the bitmap for node classification, specifying its boundary condition, node's normal direction, etc.
#### [Simulation Info](../src/CUDA/structs/simInfo.h)
Struct with simulation runtime informations, as MLUPS, bandwidth, time elapsed, etc.

## Collision Schemes

Folder with files that "implements" the collision scheme described by the file name.

## Boundary Conditions Schemes

Folder with files that declares/implements the boundary conditions schemes (e.g. pressure Zou-He, velocity Zou-He, bounce back) for the velocity sets.

## Boundary Conditions Builders

Folder with template of boundary conditions map (e.g. parallel plates using pressure condition, lid driven cavity using Zou-He) for usage/alteration.

Observations:
* For a more detailed documentations, see the respective header files.
* Instructions for use and/or code update is in the [updating](./updating.md) file.


# Post Processing

Below is described a brief of the content of the source files in _Post Processing/_ folder.

#### [File Treat](../src/Post\ Processing/fileTreat.py)
Functions to get simulation information and data (i.e. macroscopics). The configuration of the simulation files' folder is in it.

#### [Data Save](../src/Post\ Processing/dataSave.py)
Functions to save simulations' information in VTK format or csv.

#### [examplePlot](../src/Post\ Processing/exmapleVTK.py) and [exampleVTK](../src/Post\ Processing/exmaplePlot.py)
Examples of usage of the functions in the files above.

Observations:
* For more detailed documenation, see the respective files.