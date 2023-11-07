//
// Created by wxm on 2023/8/13.
//

#ifndef MULTI_GPU_COORD_WR_CUH
#define MULTI_GPU_COORD_WR_CUH
#include "Base.cuh"
int coord_WR(int N, int Dim, real height, real width, real con_x, real con_y, real var,  real size_min,
             real size_max, real *x, real *dx, real *vol);
#endif //MULTI_GPU_COORD_WR_CUH
