//
// Created by wxm on 2023/8/11.
//

#ifndef MULTI_GPU_COORD_CT_CUH
#define MULTI_GPU_COORD_CT_CUH
#include "Base.cuh"
int coord_CT_uniform(int N, real size, real width, real loc_x, real k, real d, real thick, real *x, real *dx, real *vol);
int coord_CT_ununiform(int N, real size_min, real size_max, real var, real width, real loc_x, real k, real d,
                       real area_begin, real thick, real *x, real *dx, real *vol);
#endif //MULTI_GPU_COORD_CT_CUH
