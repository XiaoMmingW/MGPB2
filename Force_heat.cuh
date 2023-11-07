//
// Created by wxm on 2023/8/13.
//

#ifndef MULTI_GPU_FORCE_HEAT_CUH
#define MULTI_GPU_FORCE_HEAT_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
#include "GPUS_Function.cuh"
void temp_surfac_correct_state(int GPUS, int MN, real k, real tempload, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                               Grid &grid, State_Thermal_Diffusion_PD2 &pd, Stream &st);
void thermal_diffusion_gpu(int GPUS, int Dim, int MN, real k, State_Thermal_Diffusion_PD2 &pd, IHP_SIZE &ihpSize,
                           Stream &st, Grid &grid);
void cal_weight_heat_gpu(int GPUS,  int Dim, int MN, int **exchange_flag, IHP_SIZE &ihpSize,
                         Grid &grid, Stream &st, State_Thermal_Diffusion_PD2 &pd);

#endif //MULTI_GPU_FORCE_HEAT_CUH
