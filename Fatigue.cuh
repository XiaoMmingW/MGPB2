//
// Created by wxm on 2023/8/11.
//

#ifndef MULTI_GPU_FATIGUE_CUH
#define MULTI_GPU_FATIGUE_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
#include "GPUS_Function.cuh"
#include "Force_state.cuh"
void fatigue(int GPUS, int NT, int MN, int break_num, real dmg_limit, real load_ratio, real A, real M, real &life_total, real &life_n,
             Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid, real *life, real &ctip);
void force_fatigue(int GPUS, int Dim, int MN, Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid);
void initial_fatigue(int GPUS, int Dim, int MN, Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid);
void initial_fatigue_state(int GPUS, int Dim, int MN, State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid);
void force_fatigue_state(int GPUS, int Dim, int MN, real K, real G, real thick, State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st,
                          Grid &grid, int **exchange_flag);
void fatigue_state(const int GPUS, int NT, int MN, int break_num, real dmg_limit, real load_ratio, real A, real M, real &life_total, real &life_n,
                   State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid, real *life, real &ctip, real *h_life);
#endif //MULTI_GPU_FATIGUE_CUH
