//
// Created by wxm on 2023/8/13.
//

#include "Coord_WR.cuh"


int coord_WR(int N, int Dim, real height, real width, real con_x, real con_y, real var,  real size_min,
             real size_max, real *x, real *dx, real *vol)
{
    int nx = (con_x-size_min/2.0)/size_min + 1;
    int ny = (con_y-size_min/2.0)/size_min + 1;
    int nz = (width-size_min/2.0)/size_min + 1;
    int count = 0;
    cout<<"nx: "<<nx<<" ny: "<<ny<<" nz: "<<nz<<endl;
    for (int k=0; k<nz; k++)
    {
        for (int j=0; j<ny; j++)
        {
            for (int i=0; i<nx; i++)
            {
                x[count] = (-nx/2.0+0.5+i)*size_min;
                x[count+N] = -(0.5+j)*size_min;
                if (Dim==3) x[count+N*2] = (-nz/2.0+0.5+k)*size_min;
                dx[count] = size_min;
                vol[count] = cubic(size_min);
                count++;
            }
        }
    }
    cout<<"cc1: "<<count<<endl;
    real tdx = size_min;
    real ty = -con_y - tdx/2.0;
    for (;ty>-height; ty-=tdx)
    {
        nx = (con_x-tdx/2.0)/tdx + 1;
        nz = (width-tdx/2.0)/tdx + 1;
        for (int j=0; j<nz; j++)
        {
            for (int i=0; i<nx; i++)
            {
                x[count] = (-nx/2.0+0.5+i)*tdx;
                x[count+N] = ty;
                if (Dim==3) x[count+N*2] = (-nz/2.0+0.5+j)*tdx;
                dx[count] = tdx;
                vol[count] = cubic(tdx);
                count++;
            }
        }
        tdx *= var;
        if (tdx>size_max) tdx = size_max;
    }

    cout<<"cc2: "<<count<<endl;
    return count;
}