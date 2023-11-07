//
// Created by wxm on 2023/8/12.
//

#ifndef MULTI_GPU_BOUNDARY_CT_CUH
#define MULTI_GPU_BOUNDARY_CT_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
#include "GPUS_Function.cuh"
void load_CT_GPU(int GPUS, real load, real width, real rad, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd, cudaStream_t *st_body);
#endif //MULTI_GPU_BOUNDARY_CT_CUH
