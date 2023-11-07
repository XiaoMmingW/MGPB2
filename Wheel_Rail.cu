//
// Created by wxm on 2023/8/13.
//

#include "Wheel_Rail.cuh"
void wheel_rail()
{
    real height = 0.002;
    int select = 6;
    real ww[8] = {0.02e-3, 0.08e-3, 0.32e-3, 1.28e-3, 5.12e-3, 10.0e-3, 20.0e-3};
    string sww[8] = {"wd0", "wd1", "wd2","wd3","wd4","wd5", "wd6"};
    real width = ww[select];
    string sw = sww[select];
    real con_x = 0.06;
    real con_y = 0.0003;
    real size_min = 0.02e-3;
    real size_max = 0.05e-3;
    real var = 1.075;
    int NI = 8000e4;
    int Dim = 3;
    if (select==0) Dim=2;
    int MN = 128;

    const int GPUS = 3;
    real E = 210.0e9;
    real pratio = 1.0/3.0;
    real k = 50.0;
    real tempload = 1.5*k;
    real cv = 450.0;
    real dt = 2.0e-7;
    int nt = 3000;
    real a = 5.88e-3;
    real b0 = 14.05e-3;
    real miu = 0.3;
    real dens = 7850.0;
    real p0 = 770.0e6;
    real v = 30.0;
    real s = 0.01;
    real vs = 1.0;//v*s;
    real xi = -con_x/2.0 + a*3.0;
    real xe = con_x/2.0 - a*3.0;
    nt = (xe-xi)/dt/v+1;
    cout<<"nt "<<nt<<endl;
    //real sedload = 0.5 * E/(1.0-pratio*pratio) * 1.0e-6;
    if (Dim==2) {
        MN = 32;
        width = size_min;
        NI = 50e4;
        tempload = k;
    }
    real *hx = new real [NI*Dim];
    real *hvol = new real [NI];
    real *hdx = new real [NI];
    int N = coord_WR(NI, Dim, height, width, con_x, con_y, var, size_min,
                  size_max, hx, hdx, hvol);
    cout<<"NT: "<<N<<endl;
    int **exchange_flag;
    array_init(0, &exchange_flag, GPUS, GPUS);
    cout<<exchange_flag[0][0]<<endl;
    real sx[GPUS];
    if (GPUS==2)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;
        sx[0] = 0.0;
    } else if (GPUS==3)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;
        exchange_flag[1][2] = exchange_flag[2][1] = 1;
        sx[0] = -con_x/6.0;
        sx[1] = con_x/6.0;
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
    data_segm(N, GPUS, horizon, sx, hx, hdx, pid, ihpSize, exchange_flag);

    isCapableP2P(GPUS);
    enableP2P (GPUS);

    State_Thermal_Diffusion_PD2 pd;
    pd.init(GPUS, Dim, MN, ihpSize.t_size);
    cout<<"ee"<<endl;

    Grid grid(GPUS, MN, ihpSize, exchange_flag);
    Stream stream(GPUS, exchange_flag);
    find_neighbor_kd3(GPUS, NI, MN, Dim,  ihpSize, pid, exchange_flag, grid,
                      hx, hdx, hvol, pd, stream.body);

    pd.init2(GPUS, Dim, MN, ihpSize.t_size);
    //vol_Corr(GPUS, Dim, MN, ihpSize, grid, pd, stream);

    //surfac_correct( GPUS, E, pratio, Dim, MN, 0, 0.6*E*1.0e-6, exchange_flag, ihpSize,grid, pd, stream);

    cal_weight_heat_gpu(GPUS, Dim, MN, exchange_flag, ihpSize, grid, stream, pd);

    temp_surfac_correct_state(GPUS, MN, k, tempload, Dim, exchange_flag, ihpSize,
            grid, pd, stream);
    real x0;
    long double start = cpuSecond();
    for (int i=0; i<nt; i++)
    {
        x0 = xi + v*i*dt;
        cout<<"time "<<i<<" "<<x0<<endl;

        move_heat_source(GPUS, Dim, x0, miu, vs, p0, b0, a, ihpSize, grid,
                pd,  stream.body);
        thermal_diffusion_gpu(GPUS, Dim, MN, k, pd, ihpSize,stream, grid);
        integrate_T_GPU(GPUS, dens, cv, dt, exchange_flag, ihpSize,grid, pd, stream);
    }
    cout<<"cost time: "<<(cpuSecond()-start)<<endl;
    if (Dim==2)
        save_T_gpu("temp_2D.txt", GPUS, Dim, ihpSize, pd);
    else
        save_T_gpu_plane("temph", GPUS, Dim, ihpSize, pd);
    //save_disp_gpu_new("disp.txt", GPUS, Dim, ihpSize, pd);

/*
    ofstream ofs;
    ofs.open("disp.txt", ios::out);

    for (int i=0; i<N; i++)
    {
        ofs<<hx[i]<<" "<<hx[i+NI]<<" "<<hx[i+NI*2]<<" "<<0.0<<" "<<0.0<<" "<<0.0<<" "<<hdx[i]<<" "<<1<<endl;
    }
    ofs.close();
    */

}