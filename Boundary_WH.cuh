//
// Created by wxm on 2023/8/14.
//

#ifndef MULTI_GPU_BOUNDARY_WH_CUH
#define MULTI_GPU_BOUNDARY_WH_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
void move_heat_source(int GPUS, int Dim, real x0, real miu, real vs, real p0, real b0, real a, IHP_SIZE &ihpSize, Grid &grid,
                      State_Thermal_Diffusion_PD2 &pd, cudaStream_t *st_body);
#endif //MULTI_GPU_BOUNDARY_WH_CUH
