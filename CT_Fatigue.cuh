//
// Created by wxm on 2023/8/11.
//

#ifndef MULTI_GPU_CT_FATIGUE_CUH
#define MULTI_GPU_CT_FATIGUE_CUH
#include "Coord.cuh"
#include "Coord_CT.cuh"
#include "Base_Function.cuh"
#include "Boundary_CT.cuh"
#include "Force_bond.cuh"
#include "Force_state.cuh"
#include "Fatigue.cuh"
#include "Integrate.cuh"
#include "Neighbor.cuh"
#include "Results.cuh"
#include "GPUS_Function.cuh"

void CT_Fatigue();
#endif //MULTI_GPU_CT_FATIGUE_CUH
