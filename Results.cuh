//
// Created by wxm on 2023/6/19.
//

#ifndef MULTI_GPU_RESULTS_CUH
#define MULTI_GPU_RESULTS_CUH
#include "Base.cuh"

void save_disp_gpu_new(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, Mech_PD &pd);
void save_T_gpu(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, State_Thermal_Diffusion_PD2 &pd);
void save_T_gpu_plane(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, State_Thermal_Diffusion_PD2 &pd);
#endif //MULTI_GPU_RESULTS_CUH
