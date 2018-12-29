//////////////////////////////////////////////////////////////////////////
// Applied High Performance Computing
//
//
// Problem : Analyze strong scaling in Shallow water equation on Open MP
//
// Method : Sixth order semi discrete finite method,
//          Fourth-order Runge-Kutta Method
//
// Author : Anubhav Singh
//
//////////////////////////////////////////////////////////////////////////

#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;

// Simulation Parameters
const int MAX_THREADS = 24;

const double x_min = 0.0;
const double x_max = 100.0;
const double y_min = 0.0;
const double y_max = 100.0;
const double t_min = 0.0;
const double t_max = 100.0;

const double Delta_x = 0.1;
const double Delta_y = 0.1;
const double Delta_t = 0.01;

const int N_x = (x_max - x_min)/Delta_x + 1;
const int N_y = (y_max - y_min)/Delta_y + 1;
const int N_t = (t_max - t_min)/Delta_t + 1;

// problem constants
const double g = 9.81; // gravitational constant

// Runge-kutta method functions
void f_vx(double** v_x, double** v_y, double** h, double** f);
void f_vy(double** v_x, double** v_y, double** h, double** f);
void f_h(double** v_x, double** v_y, double** h, double** f);

// utility functions
double** create_2darray(int m, int n);
void dump_f(fstream &file, double** arr);
void apply_periodic_boundary(double** arr);

int main(int argc, char** argv)
{

    int     i   = 0;
    int     j   = 0;
    int     l   = 0;
    int     x   = 0;
    int     y   = 0;

    fstream file;

    int grid_nx = N_x + 6;
    int grid_ny = N_y + 6;

    // Declare variables of shallow water equations
    double** h      = create_2darray(grid_nx,grid_ny);
    double** v_x    = create_2darray(grid_nx,grid_ny);
    double** v_y    = create_2darray(grid_nx,grid_ny);

    double** h_t    = create_2darray(grid_nx,grid_ny);
    double** v_x_t  = create_2darray(grid_nx,grid_ny);
    double** v_y_t  = create_2darray(grid_nx,grid_ny);

    //Declare variables for RK4 method
    double** k1_vx  = create_2darray(grid_nx,grid_ny);
    double** k2_vx  = create_2darray(grid_nx,grid_ny);
    double** k3_vx  = create_2darray(grid_nx,grid_ny);
    double** k4_vx  = create_2darray(grid_nx,grid_ny);

    double** k1_vy  = create_2darray(grid_nx,grid_ny);
    double** k2_vy  = create_2darray(grid_nx,grid_ny);
    double** k3_vy  = create_2darray(grid_nx,grid_ny);
    double** k4_vy  = create_2darray(grid_nx,grid_ny);

    double** k1_h   = create_2darray(grid_nx,grid_ny);
    double** k2_h   = create_2darray(grid_nx,grid_ny);
    double** k3_h   = create_2darray(grid_nx,grid_ny);
    double** k4_h   = create_2darray(grid_nx,grid_ny);

    double wtime    = omp_get_wtime();
    // omp_set_num_threads(MAX_THREADS);
    int N_threads   = omp_get_max_threads();


    // initialize h
    #pragma omp for schedule(static)
    for(i=3; i<N_x+3; i++)
    {
        for(j=3; j<N_y+3; j++)
        {
            h[i][j] = 1.0 + 0.5*exp(-(1.0/25.0)*(pow((x_min+(i*Delta_x))-30.0,2)
                    +pow((x_min+(j*Delta_y))-30.0,2)));
        }
    }

    // initialize h on boundary
    int end = N_y + 6;
    #pragma omp for schedule(static)
    for(int x = 0; x<N_x+6; x++)
    {
        h[x][2]=h[x][end-4];
        h[x][end-3]=h[x][3];

        h[x][1]=h[x][end-5];
        h[x][end-2]=h[x][4];

        h[x][0]=h[x][end-6];
        h[x][end-1]=h[x][5];
    }

    // initialize h on boundary
    end = N_x + 6;
    #pragma omp for schedule(static)
    for(int y = 0; y<N_y+6; y++)
    {
        h[2][y]          =   h[end-4][y];
        h[end-3][y]      =   h[3][y];

        h[1][y]          =   h[end-5][y];
        h[end-2][y]      =   h[4][y];

        h[0][y]          =   h[end-6][y];
        h[end-1][y]      =   h[5][y];
    }

    // time marching loop
    double time = 0.00;
    int int_t =0;
    char name[30];
    #pragma omp parallel default(shared) private(l, x, y)
    {
        for(l=0; l<N_t; l++)
        {
            //write to file **Uncomment to push

            //int_t = (int)(time/Delta_t);
            //#pragma omp single
            //if(int_t%(int)(1/Delta_t)==0)
            //{
            //    int int_t = (int)(time/Delta_t);
            //    sprintf(name, "assignment_1.csv.%d",int_t);
            //    file.open(name, ios::out);
            //    dump_f(file, h);
            //    file.close();
            //}

            // Calculate k1 values
            f_vx(v_x,v_y,h,k1_vx);
            f_vy(v_x,v_y,h,k1_vy);
            f_h(v_x,v_y,h,k1_h);

            #pragma omp for schedule(static)
            for(x=3; x<N_x+3; x++)
            {
                for(y=3; y<N_y+3; y++)
                {
                    v_x_t[x][y]   = v_x[x][y]   +   0.5*Delta_t*k1_vx[x][y];
                    v_y_t[x][y]   = v_y[x][y]   +   0.5*Delta_t*k1_vy[x][y];
                    h_t[x][y]     = h[x][y]     +   0.5*Delta_t*k1_h[x][y];
                }
            }
            // Message passing for sharing updated boundary point info
            apply_periodic_boundary(v_x_t);
            apply_periodic_boundary(v_y_t);
            apply_periodic_boundary(h_t);

            // Calculate k2 values
            f_vx(v_x_t,v_y_t,h_t,k2_vx);
            f_vy(v_x_t,v_y_t,h_t,k2_vy);
            f_h(v_x_t,v_y_t,h_t,k2_h);

            #pragma omp for schedule(static)
            for(x=3; x<N_x+3; x++)
            {
                for(y=3; y<N_y+3; y++)
                {
                    v_x_t[x][y]  =   v_x[x][y]  +   0.5*Delta_t*k2_vx[x][y];
                    v_y_t[x][y]  =   v_y[x][y]  +   0.5*Delta_t*k2_vy[x][y];
                    h_t[x][y]    =   h[x][y]    +   0.5*Delta_t*k2_h[x][y];
                }
            }
            // Message passing for sharing updated boundary point info
            apply_periodic_boundary(v_x_t);
            apply_periodic_boundary(v_y_t);
            apply_periodic_boundary(h_t);

            // Calculate k3 values
            f_vx(v_x_t,v_y_t,h_t, k3_vx);
            f_vy(v_x_t,v_y_t,h_t, k3_vy);
            f_h(v_x_t,v_y_t,h_t, k3_h);

            #pragma omp for schedule(static)
            for(x=3; x<N_x+3; x++)
            {
                for(y=3; y<N_y+3; y++)
                {
                    v_x_t[x][y]  =   v_x[x][y]+Delta_t*k3_vx[x][y];
                    v_y_t[x][y]  =   v_y[x][y]+Delta_t*k3_vy[x][y];
                    h_t[x][y]    =   h[x][y]+Delta_t*k3_h[x][y];
                }
            }
            // Message passing for sharing updated boundary point info
            apply_periodic_boundary(v_x_t);
            apply_periodic_boundary(v_y_t);
            apply_periodic_boundary(h_t);

            // Calculate k4 values
            f_vx(v_x_t,v_y_t,h_t, k4_vx);
            f_vy(v_x_t,v_y_t,h_t, k4_vy);
            f_h(v_x_t,v_y_t,h_t, k4_h);

            //RK4
            #pragma omp for schedule(static)
            for(x=3; x<N_x+3; x++)
            {
                for(y=3; y<N_y+3; y++)
                {
                    v_x[x][y]    =   v_x[x][y]+Delta_t*(k1_vx[x][y]/6+k2_vx[x][y]/3
                                                  +k3_vx[x][y]/3+k4_vx[x][y]/6);
                    v_y[x][y]    =   v_y[x][y]+Delta_t*(k1_vy[x][y]/6+k2_vy[x][y]/3
                                                  +k3_vy[x][y]/3+k4_vy[x][y]/6);
                    h[x][y]      =   h[x][y]+Delta_t*(k1_h[x][y]/6+k2_h[x][y]/3
                                                  +k3_h[x][y]/3+k4_h[x][y]/6);
                }
            }

            // Message passing for sharing updated boundary point info
            apply_periodic_boundary(v_x);
            apply_periodic_boundary(v_y);
            apply_periodic_boundary(h);

            #pragma omp single
            {
                time=time+Delta_t;
            }

        }
    }

    //sprintf(name, "assignment_1.csv.%d",int_t);
    //write to file ** Uncomment to push to file
    //    file.open(name, ios::out);
    //    dump_f(file, h);
    //    file.close();
    wtime   = omp_get_wtime() - wtime;  // Record the end time and calculate elapsed time
    cout << "Simulation took total =" << wtime << "and" << wtime/N_t <<
        " seconds per time step with " << N_threads << " threads" << endl;

    return 0;
}

/*
 * Utility function to create an m x n 2D array allocated on contiguous memory
 */
double** create_2darray(int m, int n)
{
    double** X = new double* [m];
    X[0] = new double[m*n];

    for(int i=1, j=n; i<m;i++, j+=n)
    {
        X[i] = &X[0][j];
        memset(X[i], 0.0, n*sizeof(double));
    }

    return X;
}

/*
 * Utility function to dump array values to file
 */
void dump_f(fstream &file, double** arr)
{
    file << "x" << "," << "y" << "," << "h" << endl;

    for(int i=3; i<N_x+3; i++)
    {
        for(int j=3; j<N_y+3; j++)
        {
           file << x_min+(i-3)*Delta_x << ","
               << y_min+(j-3)*Delta_y << ","
               << arr[i][j]  << endl;
        }
    }

}

/*
 * Utility functions to perform spatial discretization
 */
void f_vx(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    #pragma omp for nowait
    for(int x=3; x<N_x+3; x++)
    {
        for(int y=3; y<N_y+3; y++)
      {
            f[x][y] = (-v_x[x][y]*((1.0/60.0)*v_x[x+3][y]-(1.0/60.0)*v_x[x-3][y]
                            -(3.0/20.0)*v_x[x+2][y]+(3.0/20.0)*v_x[x-2][y]
                            +(3.0/4.0)*v_x[x+1][y]-(3.0/4.0)*v_x[x-1][y])/(Delta_x)
                 -v_y[x][y]*((1.0/60.0)*v_x[x][y+3]-(1.0/60.0)*v_x[x][y-3]
                            -(3.0/20.0)*v_x[x][y+2]+(3.0/20.0)*v_x[x][y-2]
                            +(3.0/4.0)*v_x[x][y+1]-(3.0/4.0)*v_x[x][y-1])/(Delta_y))
                        -g*((1.0/60.0)*h[x+3][y]-(1.0/60.0)*h[x-3][y]
                            -(3.0/20.0)*h[x+2][y]+(3.0/20.0)*h[x-2][y]
                            +(3.0/4.0)*h[x+1][y]-(3.0/4.0)*h[x-1][y])/(Delta_x);
            //if(f[x][y]!=0 && (x<5 || y<5))
                //cout<<f[x][y] <<",vxx"<<x<<",y"<<y<<endl;
        }
    }
}

/*
 * Utility functions to perform spatial discretization
 */
void f_vy(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    #pragma omp for nowait
    for(int x=3; x<N_x+3; x++)
    {
        for(int y=3; y<N_y+3; y++)
      {
            f[x][y] = (-v_x[x][y]*((1.0/60.0)*v_y[x+3][y]-(1.0/60.0)*v_y[x-3][y]
                            -(3.0/20.0)*v_y[x+2][y]+(3.0/20.0)*v_y[x-2][y]
                            +(3.0/4.0)*v_y[x+1][y]-(3.0/4.0)*v_y[x-1][y])/(Delta_x)
                   -v_y[x][y]*((1.0/60.0)*v_y[x][y+3]-(1.0/60.0)*v_y[x][y-3]
                            -(3.0/20.0)*v_y[x][y+2]+(3.0/20.0)*v_y[x][y-2]
                            +(3.0/4.0)*v_y[x][y+1]-(3.0/4.0)*v_y[x][y-1])/(Delta_y))
                         -g*((1.0/60.0)*h[x][y+3]-(1.0/60.0)*h[x][y-3]
                            -(3.0/20.0)*h[x][y+2]+(3.0/20.0)*h[x][y-2]
                            +(3.0/4.0)*h[x][y+1]-(3.0/4.0)*h[x][y-1])/(Delta_y);
            //if(f[x][y]!=0 && (x<5 || y<5))
                //cout<<f[x][y] <<",vyx"<<x<<",y"<<y<<endl;
        }
    }
}

/*
 * Utility functions to perform spatial discretization
 */
void f_h(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    #pragma omp for nowait
    for(int x=3; x<N_x+3; x++)
    {
        for(int y=3; y<N_y+3; y++)
      {
            f[x][y] = (-v_x[x][y]*((1.00/60.0)*h[x+3][y]-(1.0/60.0)*h[x-3][y]
                        -(3.0/20.0)*h[x+2][y]+(3.0/20.0)*h[x-2][y]
                        +(3.0/4.0)*h[x+1][y]-(3.0/4.0)*h[x-1][y])/(Delta_x)
              -v_y[x][y]*((1.0/60.0)*h[x][y+3]-(1.0/60.0)*h[x][y-3]
                        -(3.0/20.0)*h[x][y+2]+(3.0/20.0)*h[x][y-2]
                        +(3.0/4.0)*h[x][y+1]-(3.0/4.0)*h[x][y-1])/(Delta_y))
                -h[x][y]*(((1.0/60.0)*v_x[x+3][y]-(1.0/60.0)*v_x[x-3][y]
                        -(3.0/20.0)*v_x[x+2][y]+(3.0/20.0)*v_x[x-2][y]
                        +(3.0/4.0)*v_x[x+1][y]-(3.0/4.0)*v_x[x-1][y])/(Delta_x)
                        +((1.0/60.0)*v_y[x][y+3]-(1.0/60.0)*v_y[x][y-3]
                        -(3.0/20.0)*v_y[x][y+2]+(3.0/20.0)*v_y[x][y-2]
                        +(3.0/4.0)*v_y[x][y+1]-(3.0/4.0)*v_y[x][y-1])/(Delta_y));
            //if(f[x][y]!=0 && (x<5 || y<5))
                //cout<<f[x][y] <<",hx"<<x<<",y"<<y<<endl;
        }
    }
}

void apply_periodic_boundary(double** arr)
{
    int end = N_y + 6;
    for(int x = 0; x<N_x+6; x++)
    {
        arr[x][2]=arr[x][end-4];
        arr[x][end-3]=arr[x][3];

        arr[x][1]=arr[x][end-5];
        arr[x][end-2]=arr[x][4];

        arr[x][0]=arr[x][end-6];
        arr[x][end-1]=arr[x][5];
    }

    end = N_x + 6;
    for(int y = 0; y<N_y+6; y++)
    {
        arr[2][y]       =   arr[end-4][y];
        arr[end-3][y]   =   arr[3][y];

        arr[1][y]       =   arr[end-5][y];
        arr[end-2][y]   =   arr[4][y];

        arr[0][y]       =   arr[end-6][y];
        arr[end-1][y]   =   arr[5][y];
     }
}
