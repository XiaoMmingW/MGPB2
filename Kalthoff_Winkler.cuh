//
// Created by wxm on 2023/8/5.
//

#ifndef MULTI_GPU_KALTHOFF_WINKLER_CUH
#define MULTI_GPU_KALTHOFF_WINKLER_CUH
#include "Coord.cuh"
#include "Coord_KW.cuh"
#include "Base_Function.cuh"
#include "Boundary.cuh"
#include "Force_bond.cuh"
#include "Force_state.cuh"
#include "Integrate.cuh"
#include "Neighbor.cuh"
#include "Results.cuh"
#include "GPUS_Function.cuh"
#include "Boundary_KW.cuh"
void Kalthoff_Winkler();
#endif //MULTI_GPU_KALTHOFF_WINKLER_CUH
