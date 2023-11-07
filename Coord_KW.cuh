//
// Created by wxm on 2023/8/5.
//

#ifndef MULTI_GPU_COORD_KW_CUH
#define MULTI_GPU_COORD_KW_CUH
#include "Base.cuh"
int coord_KW(int N, int nx, int ny, int nz, real rad, real dx, real *x, real *hdx, real *hvol);

#endif //MULTI_GPU_COORD_KW_CUH
