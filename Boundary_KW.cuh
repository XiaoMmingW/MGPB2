//
// Created by wxm on 2023/8/8.
//

#ifndef MULTI_GPU_BOUNDARY_KW_CUH
#define MULTI_GPU_BOUNDARY_KW_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
struct Cylinder{
    Cylinder(int GPUS, real mass, real x, real v0): GPUS(GPUS), mass(mass), x(x), vel(v0)
    {
        acc = 0.0;
        disp = 0.0;
        force = new real* [GPUS];
        for (int i=0; i<GPUS; i++)
        {
            CHECK(cudaSetDevice(i));
            CHECK(cudaMalloc((void**)&force[i], sizeof(real)));
        }
    }
    int GPUS;
    real mass;
    real height;
    real x;
    real acc;
    real vel;
    real disp;
    real **force;
    ~Cylinder()
    {
        for (int i=0; i<GPUS; i++)
        {
            CHECK(cudaSetDevice(i));
            CHECK(cudaFree(force[i]));
        }
        delete []force;
    }
};
void set_crack_KW(int GPUS, int MN, real rad,  IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd,
                  cudaStream_t *st_body);
void cy_contact_and_integrate(int GPUS, real rad, real dt, real dens, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd,
                              cudaStream_t *st_body, Cylinder &cy);

void set_crack_KW_cpu(int N, int MN, real rad, Bond_PD_CPU &pd, int omp);
void cy_contact_and_integrate_cpu(int N, real rad, real dt, real dens, Cylinder &cy, Bond_PD_CPU &pd, int omp);

void set_crack_KW_cpu(int N, int MN, real rad, State_PD_CPU &pd, int omp);
void cy_contact_cpu(int N, real rad, real dt, real dens, real &cy_x, real &cy_disp, real &cy_force, State_PD_CPU &pd, int omp);
void cy_contact_and_integrate_cpu(int N, real rad, real dt, real dens, Cylinder &cy, State_PD_CPU &pd, int omp);
#endif //MULTI_GPU_BOUNDARY_KW_CUH
