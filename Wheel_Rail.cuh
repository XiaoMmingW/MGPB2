//
// Created by wxm on 2023/8/13.
//

#ifndef MULTI_GPU_WHEEL_RAIL_CUH
#define MULTI_GPU_WHEEL_RAIL_CUH
#include "Coord.cuh"
#include "Coord_WR.cuh"
#include "Base_Function.cuh"
#include "Boundary.cuh"
#include "Force_bond.cuh"
#include "Force_state.cuh"
#include "Force_heat.cuh"
#include "Integrate.cuh"
#include "Neighbor.cuh"
#include "Results.cuh"
#include "GPUS_Function.cuh"
#include "Boundary_WH.cuh"
void wheel_rail();
#endif //MULTI_GPU_WHEEL_RAIL_CUH
