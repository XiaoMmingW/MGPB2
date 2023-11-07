#include <iostream>
#include "ex1.cuh"
#include "ex2.cuh"
#include "Kalthoff_Winkler.cuh"
#include "CT_Fatigue.cuh"
#include "Wheel_Rail.cuh"
#include "KW_CPU.cuh"
#include "KW_State.cuh"
#include "KW_state_cpu.cuh"
struct oo{
    real x;
    real y;
};

int main() {
    //ex1();
    //ex2();
    //Kalthoff_Winkler();
    CT_Fatigue();
    //CHECK(cudaSetDevice(0));
    //wheel_rail();
//  int a = 2e7*128;
//cout<<a<<endl;
    //cout<< sizeof(long int) <<" "<< sizeof(real)<<" "<< sizeof(bool)<<endl;
    //KW_CPU();
    //KW_State();
    //KW_State_CPU();
    //PD2->free();
    //exx2<<<1,32>>>(PD);
//    int gpus;
//    cudaGetDeviceCount(&gpus);
//    cout<<"cuda capable device: "<<gpus<<endl;
    //kd_ex();
//int N = 1;
//int **xx = new int* [N];
//
//
//    int deviceCnt, dev;
//    cudaDeviceProp deviceProp;
//    dev = 2;
//    cudaSetDevice(dev);
//    cudaGetDeviceProperties(&deviceProp, dev);
//    //CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024* sizeof(int)));
//
//    printf("设备%d的内存大小：%.0f MB\n", dev, (double)deviceProp.totalGlobalMem / (1024 * 1024));
//        for (int i=0;i<N; i++)
//    CHECK(cudaMalloc((void **) &xx[i], 1024*1024*1024* sizeof(bool)*12));
//        cin>>N;
    return 0;
}
