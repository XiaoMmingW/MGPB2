//
// Created by wxm on 2023/8/14.
//

#ifndef MULTI_GPU_FORCE_BOND_CPU_CUH
#define MULTI_GPU_FORCE_BOND_CPU_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
void vol_corr_cpu(int N, int MN, int Dim, Bond_PD_CPU &pd, int omp);
void surface_correct_cpu(int N, int MN, int Dim, real E, real pratio, real sedload, real thick, Bond_PD_CPU &pd, int omp);
void bond_force_cpu(int N, int MN, const int Dim, real horizon, real pi, real sc, Bond_PD_CPU &pd, int omp);
#endif //MULTI_GPU_FORCE_BOND_CPU_CUH
