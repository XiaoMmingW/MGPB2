//
// Created by wxm on 2023/6/19.
//

#include "Base.cuh"

void array_init(int value, int ***a, int len1, int len2)
{
    *a = new int* [len1];
    for (int i=0; i<len1; i++)
    {
        (*a)[i] = new int [len2];
        for (int j=0; j<len2; j++)
        {
            (*a)[i][j] = value;
            //cout<<(*a)[i][j]<<endl;
        }
    }
}

void array_delete(int ***a, int len1)
{

    for (int i=0; i<len1; i++)
    {
        delete [](*a)[i];
    }
    delete [](*a);

}


void POINT_ID::init(int GPUS, int N, int *exchange_flag)
{
    this->GPUS = GPUS;
    this->exchange_flag = new int [GPUS];
    memcpy(this->exchange_flag, exchange_flag, GPUS* sizeof(int));
    internal = new int [N];
    halo = new int *[GPUS];
    padding = new int *[GPUS];
    cout<<this->exchange_flag[1]<<endl;
    for (int j=0; j<GPUS; j++) {
        if (exchange_flag[j]==1) {
            halo[j] = new int [N];
            padding[j] = new int [N];
        }
    }

}

void POINT_ID::Free()
{
    delete []internal;
    delete []exchange_flag;
    for (int j=0; j<GPUS; j++) {
        if (exchange_flag[j]==1) {
            delete []halo[j];
            delete []padding[j];
        }
    }
    Free_Flag = 1;
}

POINT_ID::~POINT_ID()
{
    if (Free_Flag==1){}
    else Free();
}

IHP_SIZE::IHP_SIZE(int GPUS)
{
    this->GPUS = GPUS;
    t_size = new int [GPUS];
    ih_size = new int [GPUS];
    i_size = new int [GPUS];
    array_init(0, &h_size, GPUS, GPUS);
    array_init(0, &p_size, GPUS, GPUS);
    array_init(0, &h_begin, GPUS, GPUS);
    array_init(0, &p_begin, GPUS, GPUS);
}

IHP_SIZE::~IHP_SIZE()
{
    delete []t_size;
    delete []ih_size;
    delete []i_size;
    array_delete(&h_size, GPUS);
    array_delete(&p_size, GPUS);
    array_delete(&h_begin, GPUS);
    array_delete(&p_begin, GPUS);
}




void Base_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{
    this->GPUS = GPUS;
    NN = new int * [GPUS];
    x = new real * [GPUS];
    dmg = new real * [GPUS];
    dx = new real * [GPUS];
    vol = new real * [GPUS];

    NL = new int * [GPUS];
    fail = new int * [GPUS];
    fac = new real * [GPUS];
    idist = new real * [GPUS];


    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));

        size_t size_int_point = t_size[i] * sizeof(int);
        size_t size_real_point = t_size[i] * sizeof(real);
        size_t size_int_bond = MN * t_size[i] * sizeof(int);
        size_t size_real_bond = MN * t_size[i] * sizeof(real);

        CHECK(cudaMalloc((void **) &NN[i], size_int_point));
        CHECK(cudaMalloc((void **) &dmg[i], size_real_point));
        CHECK(cudaMalloc((void **) &dx[i], size_real_point));
        CHECK(cudaMalloc((void **) &vol[i], size_real_point));
        CHECK(cudaMalloc((void **) &x[i], Dim * size_real_point));

        CHECK(cudaMalloc((void **) &NL[i], size_int_bond));
        CHECK(cudaMalloc((void **) &fail[i], size_int_bond));
        CHECK(cudaMalloc((void **) &fac[i], size_real_bond));
        CHECK(cudaMalloc((void **) &idist[i], size_real_bond));

    }
}

Base_PD::~Base_PD()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(NN[i]));
        CHECK(cudaFree(dmg[i]));
        CHECK(cudaFree(x[i]));
        CHECK(cudaFree(dx[i]));
        CHECK(cudaFree(vol[i]));
        CHECK(cudaFree(NL[i]));
        CHECK(cudaFree(fail[i]));
        CHECK(cudaFree(fac[i]));
        CHECK(cudaFree(idist[i]));
    }
    delete []NN;
    delete []dmg;
    delete []x;
    delete []dx;
    delete []vol;

    delete []NL;
    delete []fail;
    delete []fac;
    delete []idist;

}

void Mech_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{
    Base_PD::init(GPUS, Dim, MN, t_size);
    acc = new real * [GPUS];
    vel = new real * [GPUS];
    disp = new real * [GPUS];
    pforce = new real * [GPUS];
    bforce = new real * [GPUS];

    olddisp = new real * [GPUS];
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));

        size_t size_int_point = t_size[i] * sizeof(int);
        size_t size_real_point = t_size[i] * sizeof(real);
        size_t size_int_bond = MN * t_size[i] * sizeof(int);
        size_t size_real_bond = MN * t_size[i] * sizeof(real);
        CHECK(cudaMalloc((void **) &acc[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &vel[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &disp[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &pforce[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &bforce[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &olddisp[i], Dim * size_real_point));


    }
}

Mech_PD::~Mech_PD() {
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(acc[i]));
        CHECK(cudaFree(vel[i]));
        CHECK(cudaFree(disp[i]));
        CHECK(cudaFree(pforce[i]));
        CHECK(cudaFree(bforce[i]));

        CHECK(cudaFree(olddisp[i]));
    }

    delete []acc;
    delete []vel;
    delete []disp;
    delete []pforce;
    delete []bforce;
    delete []olddisp;

}

void Bond_Mech_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{

    Mech_PD::init(GPUS, Dim, MN, t_size);
    bc = new real* [GPUS];
    scr = new real* [GPUS];
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &bc[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &scr[i], MN * t_size[i] * sizeof(real)));
    }
}

Bond_Mech_PD::~Bond_Mech_PD() {
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(bc[i]));
        CHECK(cudaFree(scr[i]));
    }
    delete []bc;
    delete []scr;
}

void State_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{
    this->GPUS = GPUS;
    w = new real* [GPUS];
    theta = new real* [GPUS];
    m = new real* [GPUS];
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &w[i], MN * t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &theta[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &m[i], t_size[i] * sizeof(real)));
    }

}

State_PD::~State_PD()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(w[i]));
        CHECK(cudaFree(theta[i]));
        CHECK(cudaFree(m[i]));
    }
    delete []w;
    delete []theta;
    delete []m;
}

void State_Mech_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{

    State_PD::init(GPUS, Dim, MN, t_size);
    Mech_PD::init(GPUS, Dim, MN,  t_size);
}



void Static_PD::init(const int GPUS, int Dim, int MN, int *t_size)
{
    this->GPUS = GPUS;
    velhalfold = new real* [GPUS];
    pforceold = new real* [GPUS];
    mass = new real* [GPUS];
    cn_xy = new real* [GPUS];
    disp_xy = new real* [GPUS];
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &pforceold[i], Dim*t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &velhalfold[i], Dim*t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &mass[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &cn_xy[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &disp_xy[i], t_size[i] * sizeof(real)));
    }
}

Static_PD::~Static_PD()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaFree(pforceold[i]));
        CHECK(cudaFree(velhalfold[i]));
        CHECK(cudaFree(mass[i]));
        CHECK(cudaFree(cn_xy[i]));
        CHECK(cudaFree(disp_xy[i]));
    }
    delete []pforceold;
    delete []velhalfold;
    delete []mass;
    delete []cn_xy;
    delete []disp_xy;
}

void Fatigue_PD::init(const int GPUS, int Dim, int MN,  int *t_size)
{
    this->GPUS = GPUS;
    lambda = new real* [GPUS];
    life = new real* [GPUS];
    smax = new real* [GPUS];
    phase = new int* [GPUS];
    state = new int* [GPUS];
    c_tip = new real* [GPUS];

    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &lambda[i], MN * t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &life[i], MN*t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &smax[i], MN*t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &phase[i], t_size[i] * sizeof(int)));
        CHECK(cudaMalloc((void **) &state[i], MN*t_size[i] * sizeof(int)));
        CHECK(cudaMalloc((void **) &c_tip[i], Dim * sizeof(real)));
    }
}

Fatigue_PD::~Fatigue_PD()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(lambda[i]));
        CHECK(cudaFree(life[i]));
        CHECK(cudaFree(smax[i]));
        CHECK(cudaFree(phase[i]));
        CHECK(cudaFree(state[i]));
        CHECK(cudaFree(c_tip[i]));

    }
    delete []lambda;
    delete []life;
    delete []smax;
    delete []phase;
    delete []state;
    delete []c_tip;
}

void Bond_Fatigue_PD::init(const int GPUS, int Dim, int MN, int *t_size)
{
    Bond_Mech_PD::init(GPUS, Dim, MN, t_size);
    Fatigue_PD::init(GPUS, Dim, MN, t_size);
    Static_PD::init(GPUS, Dim, MN, t_size);
}

void State_Fatigue_PD::init(const int GPUS, int Dim, int MN, int *t_size)
{
    State_Mech_PD::init(GPUS, Dim, MN, t_size);
    Fatigue_PD::init(GPUS, Dim, MN, t_size);
    Static_PD::init(GPUS, Dim, MN, t_size);
}

void Thermal_Diffusion_PD::init(const int GPUS, int Dim, int MN, int *t_size)
{
    this->GPUS = GPUS;

    T = new real* [GPUS];
    tscr = new real* [GPUS];
    energy = new real* [GPUS];
    tbforce = new real* [GPUS];


    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &T[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &tscr[i], MN*t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &energy[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &tbforce[i], t_size[i] * sizeof(real)));
    }
}

Thermal_Diffusion_PD::~Thermal_Diffusion_PD()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(T[i]));
        CHECK(cudaFree(tscr[i]));
        CHECK(cudaFree(energy[i]));
        CHECK(cudaFree(tbforce[i]));


    }
    delete []T;
    delete []tscr;
    delete []energy;
    delete []tbforce;

}

void State_Thermal_Diffusion_PD::init(const int GPUS, int Dim, int MN, int *t_size)
{
    Base_PD::init(GPUS, Dim, MN, t_size);
    Thermal_Diffusion_PD::init(GPUS, Dim, MN, t_size);
    State_PD::init(GPUS, Dim, MN, t_size);
}

void State_Thermal_Diffusion_PD2::init(const int GPUS, int Dim, int MN,  int *t_size)
{
    this->GPUS = GPUS;
    NN = new int * [GPUS];
    x = new real * [GPUS];
    dx = new real * [GPUS];
    vol = new real * [GPUS];
    NL = new int * [GPUS];
    fail = new bool * [GPUS];

    long int tsize;
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        tsize = t_size[i];
        size_t size_int_point = tsize * sizeof(int);
        size_t size_real_point = tsize * sizeof(real);
        size_t size_int_bond = MN * tsize * sizeof(int);
        size_t size_real_bond = MN * t_size[i] * sizeof(real);
//        cout<<t_size[i]<<endl;
//        cout<<(size_int_bond/1024/1024/1024)<<endl;

        CHECK(cudaMalloc((void **) &NL[i], size_int_bond));
        CHECK(cudaMalloc((void **) &x[i], Dim * size_real_point));
        CHECK(cudaMalloc((void **) &NN[i], size_int_point));
        CHECK(cudaMalloc((void **) &dx[i], size_real_point));
        CHECK(cudaMalloc((void **) &vol[i], size_real_point));
        //CHECK(cudaMalloc((void **) &fail[i], MN * tsize * sizeof(bool)));
    }
}

void State_Thermal_Diffusion_PD2::init2(const int GPUS, int Dim, int MN,  int *t_size)
{
    this->GPUS = GPUS;

    dmg = new real * [GPUS];

    m = new real * [GPUS];
    T = new real* [GPUS];
    energy = new real* [GPUS];
    tbforce = new real* [GPUS];
    fncst = new real* [GPUS];

    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));

        size_t size_int_point = t_size[i] * sizeof(int);
        size_t size_real_point = t_size[i] * sizeof(real);
        size_t size_int_bond = MN * t_size[i] * sizeof(int);
        size_t size_real_bond = MN * t_size[i] * sizeof(real);


        CHECK(cudaMalloc((void **) &dmg[i], size_real_point));
        CHECK(cudaMalloc((void **) &m[i], t_size[i] * sizeof(real)));
        CHECK(cudaMalloc((void **) &T[i], size_real_point));
        CHECK(cudaMalloc((void **) &energy[i], size_real_point));
        CHECK(cudaMalloc((void **) &tbforce[i], size_real_point));
        CHECK(cudaMalloc((void **) &fncst[i], size_real_point));


    }
}

void State_Thermal_Diffusion_PD2::free()
{
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(T[i]));
        CHECK(cudaFree(energy[i]));
        CHECK(cudaFree(tbforce[i]));
        CHECK(cudaFree(NN[i]));
        CHECK(cudaFree(dmg[i]));
        CHECK(cudaFree(x[i]));
        CHECK(cudaFree(dx[i]));
        CHECK(cudaFree(vol[i]));
        CHECK(cudaFree(NL[i]));
        CHECK(cudaFree(fail[i]));
        CHECK(cudaFree(m[i]));
        CHECK(cudaFree(fncst[i]));


    }
    delete []T;
    delete []energy;
    delete []tbforce;
    delete []NN;
    delete []dmg;
    delete []x;
    delete []dx;
    delete []vol;
    delete []m;
    delete []NL;
    delete []fail;
    delete []fncst;

}

Stream::Stream(int GPUS, int **exchange_flag)
{
    this->GPUS = GPUS;
    halo = new cudaStream_t* [GPUS];
    body = new cudaStream_t [GPUS];
    for (int i = 0; i < GPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate( &body[i]));
        halo[i] = new cudaStream_t [GPUS];
        for (int j = 0; j < GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                CHECK(cudaStreamCreate(&halo[i][j]));
            } else
            {
                halo[i][j] = NULL;
            }
        }
    }
}

Stream::~Stream()
{
    for (int i = 0; i < GPUS; i++)
    {
        cudaStreamDestroy(body[i]);
        for (int j = 0; j < GPUS; j++) {
            if (halo[i][j] != NULL)
            {
                cudaStreamDestroy(halo[i][j]);
            }
        }
        delete []halo[i];
    }
    delete []halo;
    delete []body;
}

Grid::Grid(int GPUS, int MN, IHP_SIZE &ihpSize, int **exchange_flag)
{
    this->GPUS = GPUS;
    p_i = new int [GPUS];
    p_ih = new int [GPUS];
    p_t = new int [GPUS];
    p_h = new int *[GPUS];
    b_i = new long int [GPUS];
    b_ih = new long int [GPUS];
    b_t = new long int [GPUS];
    b_h = new long int *[GPUS];
    for (int i = 0; i < GPUS; i++)
    {

        p_i[i] = (ihpSize.i_size[i]-1)/block_size+1;
        p_ih[i] = (ihpSize.ih_size[i]-1)/block_size+1;
        p_t[i] = (ihpSize.t_size[i]-1)/block_size+1;
        b_i[i] = ((long int)ihpSize.i_size[i]*MN-1)/block_size+1;
        b_ih[i] = ((long int)ihpSize.ih_size[i]*MN-1)/block_size+1;
        b_t[i] = ((long int)ihpSize.t_size[i]*MN-1)/block_size+1;
        p_h[i] = new  int [GPUS];
        b_h[i] = new  long  int [GPUS];
        for (int j = 0; j < GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                p_h[i][j] = (ihpSize.h_size[i][j] - 1) / block_size + 1;
                b_h[i][j] = ((long int)ihpSize.h_size[i][j]*MN - 1) / block_size + 1;
            }
            else {
                p_h[i][j] = 0;
                b_h[i][j] = 0;
            }
        }
    }
}

Grid::~Grid()
{
    delete []p_i;
    delete []p_ih;
    delete []p_t;
    delete []b_i;
    delete []b_ih;
    delete []b_t;
    for (int i = 0; i < GPUS; i++)
    {
        delete []p_h[i];
        delete []b_h[i];
    }
    delete []p_h;
    delete []b_h;
}


Bond_PD_CPU::Bond_PD_CPU(int N, int MN, int Dim)
{
    //BOND
    NN = new int [N];
    dmg = new real [N];
    x = new real [N*Dim];
    dx = new real [N];
    vol = new real [N];

    NL = new int [N*MN];
    fail = new int [N*MN];
    fac = new real [N*MN];
    idist = new real [N*MN];

    bc = new real [N];
    scr = new real [N*MN];
    acc = new real [N*Dim];
    disp = new real [N*Dim];
    vel = new real [N*Dim];
    bforce = new real [N*Dim];
    pforce= new real [N*Dim];
    olddisp = new real [N*Dim];

    memset(NN, 0, N*sizeof(int));
    memset(disp, 0, N*Dim*sizeof(real));
    memset(vel, 0, N*Dim*sizeof(real));
    memset(bforce, 0, N*Dim*sizeof(real));
}

Bond_PD_CPU::~Bond_PD_CPU()
{
    delete []NN;
    delete []dmg;
    delete []x;
    delete []dx;
    delete []vol;
    delete []NL;
    delete []fail;
    delete []fac;
    delete []idist;
    delete []bc;
    delete []scr;
    delete []acc;
    delete []disp;
    delete []vel;
    delete []bforce;
    delete []pforce;
    delete []olddisp;

}

State_PD_CPU::State_PD_CPU(int N, int MN, int Dim) {

    NN = new int [N];
    dmg = new real [N];
    x = new real [N*Dim];
    dx = new real [N];
    vol = new real [N];

    NL = new int [N*MN];
    fail = new int [N*MN];
    fac = new real [N*MN];
    idist = new real [N*MN];

    m = new real [N];
    theta = new real [N];
    w = new real [N*MN];
    acc = new real [N*Dim];
    disp = new real [N*Dim];
    vel = new real [N*Dim];
    bforce = new real [N*Dim];
    pforce= new real [N*Dim];
    olddisp = new real [N*Dim];

    memset(NN, 0, N*sizeof(int));
    memset(m, 0, N*sizeof(real));
    memset(disp, 0, N*Dim*sizeof(real));
    memset(vel, 0, N*Dim*sizeof(real));
    memset(bforce, 0, N*Dim*sizeof(real));

}

State_PD_CPU::~State_PD_CPU() {
    delete []NN;
    delete []dmg;
    delete []x;
    delete []dx;
    delete []vol;
    delete []NL;
    delete []fail;
    delete []fac;
    delete []idist;
    delete []w;
    delete []theta;
    delete []m;
    delete []acc;
    delete []disp;
    delete []vel;
    delete []bforce;
    delete []pforce;
    delete []olddisp;
}