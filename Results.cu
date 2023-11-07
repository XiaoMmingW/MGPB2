//
// Created by wxm on 2023/6/19.
//

#include "Results.cuh"

void save_disp_gpu_new(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, Mech_PD &pd)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);
    for (int i=0; i<GPUS; i++)
    {

        size_t size_int = ihpSize.t_size[i]*sizeof(int);
        size_t size_real = ihpSize.t_size[i]*sizeof(real);

        real *x = new real [size_real*Dim];
        real *disp = new real [size_real*Dim];
        int *NN = new int [size_int];
        real *dx = new real [size_real];
        real *dmg = new real [size_real];
        CHECK(cudaMemcpy(x, pd.x[i], size_real*Dim, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(disp, pd.disp[i], size_real*Dim, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dx, pd.dx[i], size_real, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(NN, pd.NN[i], size_int, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dmg, pd.dmg[i], size_real, cudaMemcpyDeviceToHost));
        for (int j=0; j<ihpSize.ih_size[i]; j++)
        {
            if (Dim==2)
            {
                ofs << x[j] << " " << x[j + ihpSize.t_size[i]] << " " << disp[j] << " " << disp[j + ihpSize.t_size[i]]
                    << " " << dx[j] << " " << NN[j] <<
                    " " << dmg[j] << endl;
            } else
            {
                ofs << x[j] << " " << x[j + ihpSize.t_size[i]] <<" "<< x[j + ihpSize.t_size[i]*2] << " " << disp[j] << " " << disp[j + ihpSize.t_size[i]]
                    << " " << disp[j + ihpSize.t_size[i]*2]  << " " <<dx[j] << " " << NN[j] <<
                    " " << dmg[j] << endl;
            }
        }
        free(x);
        free(disp);
        free(dx);
        free(NN);
    }
    ofs.close();
}

void save_T_gpu(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, State_Thermal_Diffusion_PD2 &pd)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);
    for (int i=0; i<GPUS; i++)
    {

        size_t size_int = ihpSize.t_size[i]*sizeof(int);
        size_t size_real = ihpSize.t_size[i]*sizeof(real);

        real *x = new real [size_real*Dim];
        real *T = new real [size_real];
        int *NN = new int [size_int];
        real *dx = new real [size_real];
        real *dmg = new real [size_real];
        CHECK(cudaMemcpy(x, pd.x[i], size_real*Dim, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(T, pd.T[i], size_real, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dx, pd.dx[i], size_real, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(NN, pd.NN[i], size_int, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dmg, pd.dmg[i], size_real, cudaMemcpyDeviceToHost));
        for (int j=0; j<ihpSize.ih_size[i]; j++)
        {
            if (Dim==2)
            {
                ofs << x[j] << " " << x[j + ihpSize.t_size[i]] << " " << T[j]
                    << " " << dx[j] << " " << NN[j] <<
                    " " << dmg[j] << endl;
            } else
            {
                ofs << x[j] << " " << x[j + ihpSize.t_size[i]] <<" "<< x[j + ihpSize.t_size[i]*2] << " " << T[j]<<
                " " <<dx[j] << " " << NN[j] <<" " << dmg[j] << endl;
            }
        }
        free(x);
        free(T);
        free(dx);
        free(NN);
        free(dmg);
    }
    ofs.close();
}

void save_T_gpu_plane(const string FILE, int GPUS, int Dim, IHP_SIZE &ihpSize, State_Thermal_Diffusion_PD2 &pd)
{
    ofstream ofs_sf;
    ofs_sf.open(FILE+"_surface.txt", ios::out);
    ofstream ofs_st;
    ofs_st.open(FILE+"_section.txt", ios::out);

    for (int i=0; i<GPUS; i++)
    {
        size_t size_int = ihpSize.t_size[i]*sizeof(int);
        size_t size_real = ihpSize.t_size[i]*sizeof(real);

        real *x = new real [size_real*Dim];
        real *T = new real [size_real];
        int *NN = new int [size_int];
        real *dx = new real [size_real];
        real *dmg = new real [size_real];
        CHECK(cudaMemcpy(x, pd.x[i], size_real*Dim, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(T, pd.T[i], size_real, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dx, pd.dx[i], size_real, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(NN, pd.NN[i], size_int, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dmg, pd.dmg[i], size_real, cudaMemcpyDeviceToHost));
        for (int j=0; j<ihpSize.ih_size[i]; j++)
        {
            if (x[j+ihpSize.t_size[i]]>-dx[j])
            {
                ofs_sf << x[j] << " " << x[j + ihpSize.t_size[i]*2] << " " << T[j]
                    << " " << dx[j] << " " << NN[j] <<
                    " " << dmg[j] << endl;
            }

            if (fabs(x[j+ihpSize.t_size[i]*2]-dx[j]/2.0)<dx[j]/10.0)
            {
                ofs_st << x[j] << " " << x[j + ihpSize.t_size[i]] << " " << T[j]
                       << " " << dx[j] << " " << NN[j] <<
                       " " << dmg[j] << endl;
            }

        }
        free(x);
        free(T);
        free(dx);
        free(NN);
        free(dmg);
    }
    ofs_sf.close();
    ofs_st.close();
}