//
// Created by wxm on 2023/9/1.
//

#ifndef MULTI_GPU_FORCE_STATE_CPU_CUH
#define MULTI_GPU_FORCE_STATE_CPU_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
void vol_corr_cpu(int N, int MN, int Dim, State_PD_CPU &pd, int omp);
void cal_weight(
        int N, int MN, real horizon, State_PD_CPU &pd, int omp);
void cal_theta_3D(int N, int NT, int MN, State_PD_CPU &pd, int omp);
void state_force_3D(int N, real K, real G, int MN, real sc, State_PD_CPU &pd, int omp);
#endif //MULTI_GPU_FORCE_STATE_CPU_CUH
