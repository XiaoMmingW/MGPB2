//
// Created by wxm on 2023/6/19.
//

#ifndef MULTI_GPU_NEIGHBOR_CUH
#define MULTI_GPU_NEIGHBOR_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
#include "Coord.cuh"


struct kd_node
{
    int left;
    int right;
    int split;
};

void find_neighbor_kd(int GPUS, int N, int N2, int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, int* grid_point_t,
                      real *hx, real *hdx, real *hvol, Base_PD &pd, cudaStream_t *st_body);
void find_neighbor_kd2(int GPUS, int N,  int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, Grid &grid,
                       real *hx, real *hdx, real *hvol, Base_PD &pd, cudaStream_t *st_body);
void find_neighbor_kd3(int GPUS, int N, int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, Grid &grid,
                       real *hx, real *hdx, real *hvol, State_Thermal_Diffusion_PD2 &pd, cudaStream_t *st_body);
void find_neighbor_kd_cpu(int N, int MN, int Dim, Bond_PD_CPU &pd, int omp);
void find_neighbor_kd_cpu(int N, int MN, int Dim, State_PD_CPU &pd, int omp);
#endif //MULTI_GPU_NEIGHBOR_CUH

