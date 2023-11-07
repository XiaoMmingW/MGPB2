//
// Created by wxm on 2023/7/26.
//

#ifndef MULTI_GPU_GPUS_FUNCTION_CUH
#define MULTI_GPU_GPUS_FUNCTION_CUH
#include "Base.cuh"
void enableP2P (int ngpus);
bool isCapableP2P(int ngpus);

//void transfer_data(const int GPU_ID, const int GPUS, int **exchange_flag, int **halo_size, Data_GPU_Cell<TT> *dataGpuCell, cudaStream_t *st_halo);
void device_sync(int GPUS);
#endif //MULTI_GPU_GPUS_FUNCTION_CUH
