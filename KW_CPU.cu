//
// Created by wxm on 2023/8/14.
//

#include "KW_CPU.cuh"

void KW_CPU()
{
    real length = 0.2;
    real height = 0.1;
    real thick = 0.009;
    real dx = 1.0e-3;
    real E = 191.0e9;
    real pratio = 0.25;
    real dens = 8000.0;

    int nx = (length-dx/2.0)/dx+1;
    int ny = (height-dx/2.0)/dx+1;
    int nz = 9;//(thick-dx/2.0)/dx+1;
    int MN = 128;
    int NI = nx*ny*nz;
    int Dim = 3;
    int omp = 0;
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
    real *hx = new real [NI*Dim];
    real *hvol = new real [NI];
    real *hdx = new real [NI];
    int N = coord_KW(NI, nx, ny, nz, rad, dx, hx, hdx, hvol);
    Bond_PD_CPU pd(N, MN, Dim);
    memcpy(pd.x, hx, N* sizeof(real));
    memcpy(&pd.x[N], &hx[NI], N* sizeof(real));
    memcpy(&pd.x[N*2], &hx[NI*2], N* sizeof(real));
    memcpy(pd.dx, hdx, N* sizeof(real));
    memcpy(pd.vol, hvol, N* sizeof(real));
    omp_set_num_threads(160);
    Cylinder cylinder(GPUS, mass, (ny-1)/2.0 * dx +0.1*dx, v0);
    find_neighbor_kd_cpu(N, MN, Dim, pd, omp);
    cout<<"11"<<endl;
    set_crack_KW_cpu(N, MN, rad, pd, omp);
    cout<<"22"<<endl;
    long double start = cpuSecond();
    vol_corr_cpu(N, MN, Dim, pd, omp);
    cout<<"vol corr time: "<<(cpuSecond()-start)*1000<<endl;
    cpuSecond();
    surface_correct_cpu(N, MN, Dim, E, pratio, 0.6*E*1.0e-6, thick, pd, omp);
    cout<<"surf corr time: "<<(cpuSecond()-start)*1000<<endl;
    long double strat = cpuSecond();
    long double force_start, force_time=0.0, integrate_start, integrate_time=0.0, contact_start, contact_time=0.0;
    int nt2 = 50;
    for (int i=0; i<nt2; i++)
    {
        //cout<<"time "<<i<<endl;
        force_start = cpuSecond();
        memset(pd.pforce, 0, N*Dim*sizeof(real));
        bond_force_cpu(N, MN, Dim, horizon, pi, sc, pd, omp);
        force_time += cpuSecond()-force_start;
        integrate_start = cpuSecond();
        integrate_CD_cpu(N, dens, dt, Dim, pd, omp);
        integrate_time += cpuSecond()-integrate_start;
        contact_start = cpuSecond();
        cy_contact_and_integrate_cpu(N, rad, dt, dens, cylinder, pd, omp);
        contact_time += cpuSecond()-contact_start;
    }
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    start = cpuSecond();
    cal_dmg_cpu(N, MN, pd, omp);
    cout<<"dmg time: "<<(cpuSecond()-start)*1000<<endl;
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    cout<<"force time: "<<force_time*1000*nt/nt2<<endl;
    cout<<"integrate time: "<<integrate_time*1000*nt/nt2<<endl;
    cout<<"contact time: "<<contact_time*1000*nt/nt2<<endl;
    ofstream ofs;
    ofs.open("disp_bond.txt", ios::out);

    for (int i=0; i<N; i++)
    {
        ofs<<pd.x[i]<<" "<<pd.x[i+N]<<" "<<pd.x[i+N*2]<<" "<<pd.disp[i]<<" "<<pd.disp[i+N]<<" "<<pd.disp[i+N*2]
        <<" "<<hdx[i]<<" "<<pd.NN[i]<<" "<<pd.dmg[i]<<endl;
    }
    ofs.close();

}