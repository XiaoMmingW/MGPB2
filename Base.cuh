//
// Created by wxm on 2023/6/19.
//

#ifndef MULTI_GPU_BASE_CUH
#define MULTI_GPU_BASE_CUH
typedef float real;

#include "Error.cuh"
#include <math.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
using namespace std;

inline __device__ __host__ real square(real x) {return x*x;};
inline __device__ __host__ real cubic(real x) {return x*x*x;};

const real pi = acos(-1.0);
const int block_size = 128;
const int  FULL_MASK =  0xffffffff;


const real horizon = 3.015;
const real hldmg = 0.99;
__constant__ extern real ldmg;
void array_init(int value, int ***a, int len1, int len2);
void array_delete(int ***a, int len1);

struct POINT_ID
{
    void init(int GPUS,  int N=0, int *exchange_flag=NULL);

    void Free();
    int *exchange_flag;

    int Free_Flag;
    int GPUS;


    int *internal;
    int **halo;
    int **padding;
    ~POINT_ID();
};


struct IHP_SIZE
{
    IHP_SIZE(int GPUS);
    int GPUS;
    int *t_size;
    int *ih_size;
    int *i_size;
    int **h_size;
    int **p_size;
    int **h_begin;
    int **p_begin;
    ~IHP_SIZE();
};

struct Base_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);

    //ATOM
    int GPUS;

    int **NN;
    real **dmg;
    real **x;
    real **dx;
    real **vol;

    //BOND
    int **NL;
    int **fail;
    real **fac;
    real **idist;

    ~Base_PD();
};

struct Mech_PD : public Base_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);
    real **acc;
    real **disp;
    real **vel;
    real **bforce;
    real **pforce;
    real **olddisp;
    ~Mech_PD();
};


struct Bond_Mech_PD : public Mech_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);
    real **bc;
    real **scr;
    ~Bond_Mech_PD();
};

struct State_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);
    int GPUS;
    real **m;
    real **theta;
    real **w;
    ~State_PD();
};

struct State_Mech_PD : public Mech_PD, public State_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);

};


struct Static_PD
{
    void init(const int GPUS, int Dim, int MN, int *t_size);
    int GPUS;
    real **velhalfold;
    real **pforceold;
    real **mass;
    real **cn_xy;
    real **disp_xy;
    ~Static_PD();
};

struct Fatigue_PD
{
    void init(const int GPUS, int Dim, int MN, int *t_size);
    int GPUS;
    int **phase;
    real **lambda;
    real **life;
    real **smax;
    int **state;
    real **c_tip;
    ~Fatigue_PD();
};
struct Bond_Fatigue_PD : public Bond_Mech_PD, public Fatigue_PD, public Static_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);
};

struct State_Fatigue_PD : public State_Mech_PD, public Fatigue_PD, public Static_PD
{
    void init(const int GPUS, int Dim, int MN,  int *t_size);
};



struct Thermal_Diffusion_PD
{
    void init(const int GPUS, int Dim, int MN, int *t_size);
    int GPUS;
    real **T;
    real **tscr;
    real **energy;
    real **tbforce;
    ~Thermal_Diffusion_PD();
};

struct State_Thermal_Diffusion_PD : public Base_PD, public Thermal_Diffusion_PD, public State_PD
{
    void init(const int GPUS, int Dim, int MN, int *t_size);
};

struct State_Thermal_Diffusion_PD2
{
    void init(const int GPUS, int Dim, int MN, int *t_size);
    void init2(const int GPUS, int Dim, int MN, int *t_size);
    //ATOM
    int GPUS;

    int **NN;
    real **dmg;
    real **x;
    real **dx;
    real **vol;
    real **T;
    real **fncst;

    real **energy;
    real **tbforce;
    real **m;

    //BOND
    int **NL;
    bool **fail;


    void free();

};

struct Point_ID_CPU
{
    void init(int N, int GPUS, int **exchange_number);
    int GPUS;
    int **exchange_flag;
    int **internal;
    int *i_size;
    int ***halo;
    int **h_size;
    int ***padding;
    int **p_size;
    ~Point_ID_CPU();
};


struct Stream
{
    Stream(int GPUS, int **exchange_flag);
    int GPUS;
    cudaStream_t** halo;
    cudaStream_t* body;
    ~Stream();
};


struct Grid
{
    Grid(int GPUS, int MN, IHP_SIZE &ihpSize, int **exchange_flag);
    int GPUS;
    int* p_i;
    int* p_ih;
    int* p_t;
    int** p_h;
    long int* b_i;
    long int* b_ih;
    long int* b_t;
    long int **b_h;
    ~Grid();
};

struct Bond_PD_CPU
{
    Bond_PD_CPU(int N, int MN, int Dim);
    int *NN;
    real *dmg;
    real *x;
    real *dx;
    real *vol;

    //BOND
    int *NL;
    int *fail;
    real *fac;
    real *idist;

    real *bc;
    real *scr;

    real *acc;
    real *disp;
    real *vel;
    real *bforce;
    real *pforce;
    real *olddisp;
    real *oldvel;
    ~Bond_PD_CPU();
};

struct State_PD_CPU
{
    State_PD_CPU(int N, int MN, int Dim);
    int *NN;
    real *dmg;
    real *x;
    real *dx;
    real *vol;

    //BOND
    int *NL;
    int *fail;
    real *fac;
    real *idist;

    real *w;
    real *m;
    real *theta;


    real *acc;
    real *disp;
    real *vel;
    real *bforce;
    real *pforce;
    real *olddisp;
    real *oldvel;
    ~State_PD_CPU();
};

#endif //MULTI_GPU_BASE_CUH
