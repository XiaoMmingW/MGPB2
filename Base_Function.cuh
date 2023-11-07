//
// Created by wxm on 2023/7/26.
//

#ifndef MULTI_GPU_BASE_FUNCTION_CUH
#define MULTI_GPU_BASE_FUNCTION_CUH
#include "Base.cuh"
#include "GPUS_Function.cuh"
__device__ real atomicAdd2(real* address, real val);
long double cpuSecond();
__global__ void kernel_cal_dmg_2D(int N, int MN, int *NN,real *dmg, real *vol, int *NL,int *fail, real *fac);
__global__ void kernel_initial_fail(int N, int MN, int *fail);
__global__ void gpu_set_crack(
        int N, int MN, real pi, real clength, real loc_x, real loc_y,real theta, int rint, int *g_NN, int *g_fail,
        int *g_NL, real *x);
__device__ __host__ real cal_dist( int NT, int id1, int id2, real *x, int Dim);
__global__ void kernel_vol_Corr
        (int N, int MN, real horizon, int Dim, int *NN, int *NL, real *x, real *dx, real *idist, real *fac);
void vol_Corr(int GPUS, int Dim, int MN, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st);
void set_crack_2D(int GPUS, int MN, real clength, real loc_x, real loc_y, real theta, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st);
void cal_dmg_gpu(int GPUS, int MN, int Dim, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st, int **exchange_flag);

void cal_mass_GPU(int GPUS, int Dim, real E, real pratio, real size_min, real thick, IHP_SIZE &ihpSize, Grid &grid,
                  Static_PD &pd, cudaStream_t *st_body);
void cal_dmg_cpu(int N, int MN, Bond_PD_CPU &pd, int omp);
void cal_dmg_cpu(int N, int MN, State_PD_CPU &pd, int omp);
#endif //MULTI_GPU_BASE_FUNCTION_CUH
