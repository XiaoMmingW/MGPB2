//
// Created by wxm on 2023/8/5.
//

#include "Kalthoff_Winkler.cuh"
void Kalthoff_Winkler()
{
    real length = 0.2;
    real height = 0.1;
    real thick = 0.009;
    real var = pow(3.0,1.0/3.0);
    real dx = 1.0e-3;

    real E = 191.0e9;
    real pratio = 0.25;
    real dens = 8000.0;

    int nx = (length-dx/2.0)/dx+1;
    int ny = (height-dx/2.0)/dx+1;
    int nz = (thick-dx/2.0)/dx+1;
    int MN = 128;
    int N = nx*ny*nz;
    int Dim = 3;
    real a0 = 0.5;
    real h0 = 1.5e-3;
    real rad = 0.025;
    real v0 = -32.0;
    real mass = 1.57;
    real dt = 8.7e-8;
    int nt = 1350;
    real sc = 0.01;
    const int GPUS = 3;
    cout<<nz<<endl;
    real *hx = new real [N*Dim];
    real *hvol = new real [N];
    real *hdx = new real [N];
    N = coord_KW(N, nx, ny, nz, rad, dx, hx, hdx, hvol);
    cout<<"total N: "<<N<<endl;
//    ofstream ofs;
//    ofs.open("disp.txt", ios::out);
//    cout<<N<<endl;
//    for (int i = 0; i < N/9; i++) {
//        //for (int j=0; j<ihpSize.ih_size[i]; j++)
//        {
//            ofs<<hx[i]<<" "<<hx[i+nx*ny*nz]<<" "<<0.0<<" "<<0.0<<" "<<dx<<" "<<0.0<<endl;
//
//        }
//    }
//
//    ofs.close();

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
        sx[0] = -length/6.0;
        sx[1] = length/6.0;

    }

    POINT_ID pid[GPUS];

    IHP_SIZE ihpSize(GPUS);

    //#pragma omp parallel for
    for (int i=0;i<GPUS;i++)
    {
        pid[i].init(GPUS, N, exchange_flag[i]);
    }


    data_segm(N, GPUS, horizon, sx, hx, hdx, pid, ihpSize, exchange_flag);

//    for (int i = 0; i < GPUS; i++) {
//        cout<<"i_size "<<ihpSize.i_size[i]<<" t_size "<<ihpSize.t_size[i]<<" ih_size "<<ihpSize.ih_size[i]<<endl;
//        for (int j = 0; j < GPUS; j++) {
//            if (exchange_flag[i][j]==1)
//            {
//                cout<<"h_begin "<<ihpSize.h_begin[i][j]<<" p_begin "<<ihpSize.p_begin[i][j]<<endl;
//            }
//        }
//    }


    isCapableP2P(GPUS);
    enableP2P (GPUS);
    //多少个GPU生成多少数据
    cout<<"ee"<<endl;
    Cylinder cylinder(GPUS, mass, (ny-1)/2.0 * dx +0.1*dx, v0);
    cout<<"ee"<<endl;
    Bond_Mech_PD pd;
    pd.init(GPUS, Dim, MN, ihpSize.t_size);
    cout<<"ee"<<endl;
    Grid grid(GPUS, MN, ihpSize, exchange_flag);
    Stream stream(GPUS, exchange_flag);
    find_neighbor_kd2(GPUS, nx*ny*nz, MN, Dim,  ihpSize, pid, exchange_flag, grid,
            hx, hdx, hvol, pd, stream.body);
    long double start = cpuSecond();
    set_crack_KW(GPUS, MN, rad, ihpSize, grid, pd, stream.body);
    cout<<"set crack time： "<<(cpuSecond()-start)*1000<<endl;
    start = cpuSecond();
    vol_Corr(GPUS, Dim, MN, ihpSize, grid, pd, stream);
    cout<<"vol_Corr time： "<<(cpuSecond()-start)*1000<<endl;
    start = cpuSecond();
    surfac_correct(GPUS, E, pratio, Dim, MN, 0, 0.6*E*1.0e-6, exchange_flag, ihpSize,grid, pd, stream);
    cout<<"surface_Corr time： "<<(cpuSecond()-start)*1000<<endl;
    long double force_start, force_time=0.0, integrate_start, integrate_time=0.0, contact_start, contact_time=0.0;
    long double strat = cpuSecond();
    int nt2 = 400;
    for (int i=0; i<nt2; i++)
    {
        //cout<<"time "<<i<<endl;
        force_start = cpuSecond();
        force_gpu(GPUS, sc, Dim, MN, pd, ihpSize, stream, grid);
        force_time += cpuSecond()-force_start;
        integrate_start = cpuSecond();
        integrate_CD_contact(GPUS, dens, dt, Dim, exchange_flag, ihpSize,grid, pd, stream);
        integrate_time += cpuSecond()-integrate_start;
        contact_start = cpuSecond();
        cy_contact_and_integrate(GPUS, rad, dt, dens, ihpSize, grid, pd,stream.body, cylinder);
        contact_time += cpuSecond()-contact_start;
    }
    start = cpuSecond();
    cal_dmg_gpu(GPUS, MN, Dim, ihpSize, grid, pd, stream, exchange_flag);
    cout<<"dmg time: "<<(cpuSecond()-start)*1000<<endl;
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    cout<<"force time: "<<force_time*1000*nt/nt2<<endl;
    cout<<"integrate time: "<<integrate_time*1000*nt/nt2<<endl;
    cout<<"contact time: "<<contact_time*1000*nt/nt2<<endl;
    save_disp_gpu_new("disp_double.txt", GPUS, Dim, ihpSize, pd);
}