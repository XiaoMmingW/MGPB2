//
// Created by wxm on 2023/8/11.
//

#include "Coord_CT.cuh"

int coord_CT_uniform(int N, real size, real width, real loc_x, real k, real d, real thick, real *x, real *dx, real *vol)
{
    int nx = (1.25*width-size/2)/size + 1;
    int ny = (1.2*width/2.0 -size/2)/size + 1;
    cout<<"nx "<<nx<<" "<<ny<<endl;
    //生成裂纹扩展区点
    real tx;
    real ty;
    int num=0;
    for (int i=0; i<ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            tx = (0.5 + j) * size;
            ty = (0.5 + i) * size;
            if (ty<=d/2.0 and ty>=k*(tx-loc_x))
            {
                x[num] = tx;
                x[num + N] = ty;
                dx[num] = size;
                vol[num] = size*size*thick;
                num++;
            } else if (ty>d/2.0)
            {
                x[num] = tx;
                x[num + N] = ty;
                dx[num] = size;
                vol[num] = size*size*thick;
                num++;
            }
        }
    }
    cout<<"num "<<num<<endl;

    for (int i=0; i<num; i++)
    {
        x[num+i] = x[i];
        x[N+num+i] = -x[N+i];
        dx[num+i] = size;
        vol[num+i] = size*size*thick;
    }
    return num*2;
}

int coord_CT_ununiform(int N, real size_min, real size_max, real var, real width, real loc_x, real k, real d,
                       real area_begin, real thick, real *x, real *dx, real *vol)
{
    real tdx = size_min;
    int nx;
    real tx;
    real ty=tdx/2.0;
    int num=0;
    for (; ty<1.2*width/2.0-size_max/12.0; ty+=tdx)
    {
        nx = (1.25*width-tdx/2)/tdx + 1;
        for (int j = 0; j < nx; j++)
        {
            tx = (0.5 + j) * tdx;
            if (ty<=d/2.0 and ty>=k*(tx-loc_x))
            {
                x[num] = tx;
                x[num + N] = ty;
                dx[num] = tdx;
                vol[num] = tdx*tdx*thick;
                num++;
            } else if (ty>d/2.0)
            {
                x[num] = tx;
                x[num + N] = ty;
                dx[num] = tdx;
                vol[num] = tdx*tdx*thick;
                num++;
            }
        }
        if (ty>area_begin)
        {
            tdx*=var;
            if (tdx>size_max) tdx = size_max;
        }
    }
    cout<<"num "<<num<<endl;

    for (int i=0; i<num; i++)
    {
        x[num+i] = x[i];
        x[N+num+i] = -x[N+i];
        dx[num+i] = dx[i];
        vol[num+i] = vol[i];
    }

    return num*2;
}


