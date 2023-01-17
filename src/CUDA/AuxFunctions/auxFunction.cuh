
#ifndef __AUX_FUNCTIONS_H
#define __AUX_FUNCTIONS_H

#include "reduction.cuh"
#include "./../var.h"
#include "./../structs/macroscopics.h"


__host__
dfloat mean_macro(Macroscopics const macr, int macro_index, size_t step);

#endif