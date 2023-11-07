//
// Created by wxm on 2023/6/19.
//

#ifndef MULTI_GPU_BOUNDARY_CUH
#define MULTI_GPU_BOUNDARY_CUH
#include "Base.cuh"
__global__ void load_vel(real *disp, real *x, real ct, int N, real vel_load, real load_area);
__global__ void load_vel_new(real *disp, real *x, real ct, int N, int NT, real vel_load, real load_area);
void load_vel(int GPUS,  real ct, real vel_load, real load_area, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd, Stream &st);
#endif //MULTI_GPU_BOUNDARY_CUH
