//
// Created by wxm on 2023/7/24.
//

#include "ex2.cuh"

void transfer(Base_PD &pd)
{
    real *x = new real [10];
    x[0] =100;
    CHECK(cudaMemcpy(pd.x, x, 10*sizeof(real), cudaMemcpyHostToDevice));
}



void ex2()
{

    real length = 0.05;
    real height = 0.05;
    real dx = 0.1e-3;
    int nx = (length-dx/2.0)/dx+1;
    int ny = (height-dx/2.0)/dx+1+6;
    int MN = 32;
    int N = nx*ny;
    int Dim = 2;

    int nt = 1000;
    real dt = 1.3367e-8;
    real sc = 0.04472;
    real E = 192.0e9;
    real pratio = 1.0/3.0;
    real K = E/3.0/(1.0-2.0*pratio);
    real G = 0.5*E/(1.0+pratio);
    real dens = 8000.0;
    real vel_load = 50.0;
    real *hx = new real [N*Dim];
    real *hvol = new real [N];
    real *hdx = new real [N];
    coord_plate(hx, hdx, hvol, N, nx, ny, dx, cubic(dx));

    const int GPUS = 3;
    int segm_num = GPUS-1;

//    CHECK(cudaMemcpy(coord.x, hx, N * sizeof(real), cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(&coord.x[N], &hx[N], N* sizeof(real), cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(coord.vol, hvol, N* sizeof(real), cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(coord.dx, hdx, N* sizeof(real), cudaMemcpyHostToDevice));
    int **exchange_flag;
    array_init(0, &exchange_flag, GPUS, GPUS);
    cout<<exchange_flag[0][0]<<endl;
    if (GPUS==2)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;

    } else if (GPUS==3)
    {
        exchange_flag[0][1] = exchange_flag[1][0] = 1;
        exchange_flag[1][2] = exchange_flag[2][1] = 1;

    }

    POINT_ID pid[GPUS];

    IHP_SIZE ihpSize(GPUS);

    for (int i=0;i<GPUS;i++)
    {
        pid[i].init(GPUS, N, exchange_flag[i]);
    }

    for (int i=0;i<GPUS;i++)
    {

    }
    if (segm_num==0) segm_num=1;
    real k[segm_num], b[segm_num];
    if (GPUS==1)
    {
        //ihpSize.i_size[0] = ihpSize.t_size[0] = ihpSize.ih_size[0] = N;

    } else
    {

        if (GPUS == 2) {
            k[0] = 0.0;
            b[0] = 0.0;
        } else if (GPUS == 3) {
            k[0] = 0.0;
            k[1] = 0.0;
            b[0] = height / 6.0;
            b[1] = -height / 6.0;
        }

    }
    data_segm_2D(N, GPUS, segm_num, k, b, hx, hdx, pid, ihpSize, exchange_flag);
    //pid[0].halo[1][0] = 0;
    for (int i = 0; i < GPUS; i++) {
        cout<<"i_size "<<ihpSize.i_size[i]<<" t_size "<<ihpSize.t_size[i]<<" ih_size "<<ihpSize.ih_size[i]<<endl;
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j]==1)
            {
                cout<<"h_begin "<<ihpSize.h_begin[i][j]<<" p_begin "<<ihpSize.p_begin[i][j]<<endl;
            }
        }
    }


    isCapableP2P(GPUS);
    enableP2P (GPUS);
    //多少个GPU生成多少数据

    Bond_Mech_PD pd;
    pd.init(GPUS, Dim, MN, ihpSize.t_size);
    Grid grid(GPUS, MN, ihpSize, exchange_flag);
    Stream stream(GPUS, exchange_flag);
    find_neighbor_kd2(GPUS, N, MN, Dim,  ihpSize, pid, exchange_flag, grid,
                      hx, hdx, hvol, pd, stream.body);


    device_sync(GPUS);
    vol_Corr(GPUS, Dim, MN, ihpSize, grid, pd, stream);
    surfac_correct( GPUS, E, pratio, Dim, MN, dx, 0.5 * E/(1.0-pratio*pratio) * 1.0e-6, exchange_flag, ihpSize,grid, pd, stream);
    set_crack_2D(GPUS, MN, 0.01, 0.0, 0.0, pi, ihpSize, grid, pd, stream);
    long double strat = cpuSecond();
    for (int i=0; i<nt; i++)
    {
        cout<<"time "<<i<<endl;
        //integrate_1(GPUS,  dt, Dim, exchange_flag, ihpSize,grid, pd, stream);

        force_gpu(GPUS, sc, Dim, MN, pd, ihpSize, stream, grid);
        //integrate_2(GPUS, dens, dt, Dim, ihpSize, grid, pd, stream);
        integrate_CD(GPUS, dens, dt, Dim, exchange_flag, ihpSize,grid, pd, stream);
        load_vel(GPUS,  i*dt, vel_load, height/2.0, ihpSize, grid, pd, stream);
    }
    cal_dmg_gpu(GPUS, MN, Dim, ihpSize, grid, pd, stream, exchange_flag);
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    save_disp_gpu_new("disp.txt", GPUS, Dim, ihpSize, pd);
}