//
// Created by wxm on 2023/8/11.
//

#include "CT_Fatigue.cuh"
void CT_Fatigue()
{
    real width = 50.0e-3;
    real rad = 0.25*width/2.0;
    real size_min = 0.3e-3;
    real size_max = 0.5e-3;
    real var = 1.05;
    real thick = 0.01;
    real d = 0.0015;
    real q = 10.5e-3;
    int MN = 32;
    int NI = 50e4;
    int Dim = 2;
    const int GPUS = 3;
    real E = 210.0e9;
    real pratio = 0.3;
    real K = E/3.0/(1.0-2*pratio);
    real G = 0.5*E*(1.0+pratio);
    real A = 433.31, M = 3.3595;
    real dt = 1.0;
    int nt = 4000;
    int break_num = 4;
    real dmg_limit = 0.38;
    real load_ratio = 0.1;
    real load_force = 11.5e3*5.0*1.0;
    real sedload = 0.5 * E/(1.0-pratio*pratio) * 1.0e-6;

    real load = load_force/(pi*rad*rad*thick);
    real *hx = new real [NI*Dim];
    real *hvol = new real [NI];
    real *hdx = new real [NI];
    //int N = coord_CT_uniform(NI, size_min, width, width-7.5e-3, 1/sqrt(3.0), d, thick, hx, hdx, hvol);
    int N = coord_CT_ununiform(NI, size_min, size_max, var, width, width-7.5e-3, 1/sqrt(3.0), d,
           width/2.0/5.0*(7.0), thick, hx, hdx, hvol);
    cout<<"total point: "<<N<<endl;
    int **exchange_flag;
    array_init(0, &exchange_flag, GPUS, GPUS);
    cout<<exchange_flag[0][0]<<endl;
    real sx[GPUS];
    if (GPUS==2)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;
        sx[0] = 1.25*width/2.0;
    } else if (GPUS==3)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;
        exchange_flag[1][2] = exchange_flag[2][1] = 1;
        sx[0] = 1.25*width/3.0;
        sx[1] = 2.0*1.25*width/3.0;
    }

    POINT_ID pid[GPUS];
    IHP_SIZE ihpSize(GPUS);
    real life_total = 0.0;
    real life_n = 0.0;
    //#pragma omp parallel for
    for (int i=0;i<GPUS;i++)
    {
        pid[i].init(GPUS, N, exchange_flag[i]);
    }

    data_segm(N, GPUS, horizon+2.0, sx, hx, hdx, pid, ihpSize, exchange_flag);
    isCapableP2P(GPUS);
    enableP2P (GPUS);
    State_Fatigue_PD pd;
    pd.init(GPUS, Dim, MN, ihpSize.t_size);

    Grid grid(GPUS, MN, ihpSize, exchange_flag);
    Stream stream(GPUS, exchange_flag);
    find_neighbor_kd2(GPUS, NI, MN, Dim,  ihpSize, pid, exchange_flag, grid,
                      hx, hdx, hvol, pd, stream.body);
    device_sync(GPUS);
    vol_Corr(GPUS, Dim, MN, ihpSize, grid, pd, stream);
    //surfac_correct( GPUS, E, pratio, Dim, MN, thick, sedload, exchange_flag, ihpSize,grid, pd, stream);
    load_CT_GPU(GPUS, load, width, rad, ihpSize, grid, pd, stream.body);
    cal_mass_GPU(GPUS, Dim, E, pratio, size_min, thick, ihpSize, grid,pd, stream.body);
    cal_weight_gpu(GPUS,  Dim, MN, exchange_flag, ihpSize,grid, stream, pd, pd);
    real clength = (0.25*width+q)*1.1;
    real loc_x = clength/2.0+width-q;
    set_crack_2D(GPUS, MN, 0.027*2.0, width, 0.0, pi, ihpSize, grid, pd, stream);
    set_crack_2D(GPUS, MN, width*0.5, 1.25*width, 0.0, pi, ihpSize, grid, pd, stream);
    real *life;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc((void **) &life, N*MN*sizeof(real)));
    cal_dmg_gpu(GPUS, MN, Dim, ihpSize, grid, pd, stream, exchange_flag);
    real dd;
    long double start = cpuSecond();
    ofstream ofs;
    ofs.open("his_t01.txt", ios::out);
    real c_tip = width*2.0;
    real *h_life = new real [break_num*GPUS];
    for (int j=0; j<0; j++)
    {
        cout<<"time: "<<j<<endl;
        initial_fatigue_state(GPUS, Dim, MN, pd, ihpSize, stream, grid);
        for (int i=0; i<nt; i++)
        {
            force_fatigue_state(GPUS, Dim, MN, K, G, thick, pd, ihpSize, stream, grid, exchange_flag);
            static_integrate(GPUS, Dim, dt, i, exchange_flag, pd, ihpSize, stream, grid);
            //CHECK(cudaMemcpy(&dd, &pd.disp[2][ihpSize.ih_size[2]-1+ihpSize.t_size[2]], sizeof(real), cudaMemcpyDeviceToHost));
        }
        fatigue_state(GPUS, N, MN, break_num, dmg_limit, load_ratio, A, M, life_total, life_n,
                pd, ihpSize, stream, grid, life, c_tip, h_life);
        ofs<<life_total<<" "<<(width-c_tip)*1000<<endl;
        cout<<"crack length "<<((width)-c_tip)*1000<<endl;

        cal_dmg_gpu(GPUS, MN, Dim, ihpSize, grid, pd, stream, exchange_flag);
        cout<<"life_total: "<<life_total<<" life_n: "<<life_n<<endl;
        if (((width)-c_tip)*1000>=15.0) break;
    }
    ofs.close();
    cout<<"cost time "<<(cpuSecond()-start)*160.0<<endl;
    save_disp_gpu_new("disp_t06.txt", GPUS, Dim, ihpSize, pd);


/*
    ofstream ofs;
    ofs.open("disp.txt", ios::out);
    for (int i=0; i<N; i++)
    {
        ofs<<hx[i]<<" "<<hx[i+NI]<<" "<<0.0<<" "<<0.0<<" "<<hdx[i]<<" "<<1<<endl;
    }
    ofs.close();
    */
}