//////////////////////////////////////////////////////////////////////////
// Applied High Performance Computing
//
//
// Problem : Analyze Shallow water equation weak scaling using MPI
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
#include <mpi.h>

using namespace std;

// MPI Global Settings

const int N_D                   = 2;
const int X                     = 0;
const int Y                     = 1;
const int numElementsPerBlock   = 1;

// Simulation Parameters

const double x_min = 0.0;
const double x_max = 100.0;
const double y_min = 0.0;
const double y_max = 100.0;
const double t_min = 0.0;
const double t_max = 100.0;
// Fixing the problem size per processor
const int myN_x = 50;
const int myN_y = 50;
int N_x;
int N_y;
double Delta_x = 1;
double Delta_y = 1;
const double Delta_t = 0.01;

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
// Exchange the boundary grid-point values with neighbors
void exchange(double** phi, int myN_x, int myN_y, int leftNeighbor, int rightNeighbor, int bottomNeighbor, int topNeighbor, MPI_Comm Comm2D, MPI_Status* status, MPI_Datatype strideType);

int main(int argc, char** argv)
{

    int     l   = 0;

    int myID;
    int N_Procs;

    fstream file;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N_Procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    int N                   = sqrt(N_Procs);
    int dimensions[N_D]     = {N, N};
    int isPeriodic[N_D]     = {1,1};
    int myCoords[N_D]       = {0,0};
    N_x			    = myN_x*dimensions[X]+1;
    N_y			    = myN_y*dimensions[Y]+1;
    Delta_x		    = (x_max-x_min)/(N_x-1);
    Delta_y		    = (y_max-y_min)/(N_y-1);
    int myiStart            = 0;
    int myiEnd              = 0;
    int myjStart            = 0;
    int myjEnd              = 0;
    double psi              = 0;
    double Delta_xy         = Delta_x;
    double myr_norm         = 0.0;
    double r_norm           = 0.0;
    double tolerance        = 1e-6;
    int N_k                 = 1000000;
    int i                   = 0;
    int j                   = 0;
    int k                   = 0;
    int reorder             = 1;
    int leftNeighbor        = 0;
    int rightNeighbor       = 0;
    int bottomNeighbor      = 0;
    int topNeighbor         = 0;
    double wtime            = 0;
    char myFileName[64];
    MPI_Status status;
    MPI_Datatype strideType;
    MPI_Comm Comm2D;


    int grid_nx = myN_x + 6;
    int grid_ny = myN_y + 6;

    double** h      = create_2darray(grid_nx,grid_ny);
    double** v_x    = create_2darray(grid_nx,grid_ny);
    double** v_y    = create_2darray(grid_nx,grid_ny);

    double** h_t    = create_2darray(grid_nx,grid_ny);
    double** v_x_t  = create_2darray(grid_nx,grid_ny);
    double** v_y_t  = create_2darray(grid_nx,grid_ny);

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

    MPI_Cart_create(MPI_COMM_WORLD, N_D, dimensions, isPeriodic, reorder, &Comm2D);
    MPI_Comm_rank(Comm2D, &myID);
    MPI_Cart_coords(Comm2D, myID, N_D, myCoords);
    MPI_Cart_shift(Comm2D, X, 1, &leftNeighbor, &rightNeighbor);
    MPI_Cart_shift(Comm2D, Y, 1, &bottomNeighbor, &topNeighbor);

    myiStart    = myCoords[X]==0    ? 2     : 1;
    myiEnd      = myCoords[X]==N-1  ? myN_x : myN_x+1;
    myjStart    = myCoords[Y]==0    ? 2     : 1;
    myjEnd      = myCoords[Y]==N-1  ? myN_y : myN_y+1;


    MPI_Type_vector(myN_x, numElementsPerBlock, myN_y+6, MPI_DOUBLE, &strideType);
    MPI_Type_commit(&strideType);

    if(myID==0)
    {
        wtime = MPI_Wtime();
    }

    // initialize h
    for(i=3; i<myN_x+3; i++)
    {
        for(j=3; j<myN_y+3; j++)
        {
            int ii = myCoords[X]*myN_x + i - 1;
            int jj = myCoords[Y]*myN_y + j - 1;
            h[i][j] = 1.0 + 0.5*exp(-(1.0/25.0)*(pow((x_min+(ii*Delta_x))-30.0,2)
                        +pow((x_min+(jj*Delta_y))-30.0,2)));
        }
    }


    // Runge-Kutta time marching loop

    double time = 0.00;
    char name[30];
    for(l=0; l<N_t; l++)
    {

        int int_t = (int)(time/Delta_t);

        //sprintf(name, "assignment_1.csv.%d_%d_%d",int_t, myCoords[X], myCoords[Y]);

        //write to file
        //file.open(name, ios::out);
        //dump_f(file, h);

        exchange(v_x, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(v_y, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(h, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);

        f_vx(v_x,v_y,h,k1_vx);
        f_vy(v_x,v_y,h,k1_vy);
        f_h(v_x,v_y,h,k1_h);


        for(int x=3; x<myN_x+3; x++)
        {
            for(int y=3; y<myN_y+3; y++)
            {
                v_x_t[x][y]   = v_x[x][y]   +   0.5*Delta_t*k1_vx[x][y];
                v_y_t[x][y]   = v_y[x][y]   +   0.5*Delta_t*k1_vy[x][y];
                h_t[x][y]     = h[x][y]     +   0.5*Delta_t*k1_h[x][y];
            }
        }

        exchange(v_x_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(v_y_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(h_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);

        f_vx(v_x_t,v_y_t,h_t,k2_vx);
        f_vy(v_x_t,v_y_t,h_t,k2_vy);
        f_h(v_x_t,v_y_t,h_t,k2_h);

        for(int x=3; x<myN_x+3; x++)
        {
            for(int y=3; y<myN_y+3; y++)
            {
                v_x_t[x][y]  =   v_x[x][y]  +   0.5*Delta_t*k2_vx[x][y];
                v_y_t[x][y]  =   v_y[x][y]  +   0.5*Delta_t*k2_vy[x][y];
                h_t[x][y]    =   h[x][y]    +   0.5*Delta_t*k2_h[x][y];
            }
        }

        exchange(v_x_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(v_y_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(h_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);

        f_vx(v_x_t,v_y_t,h_t, k3_vx);
        f_vy(v_x_t,v_y_t,h_t, k3_vy);
        f_h(v_x_t,v_y_t,h_t, k3_h);

        for(int x=3; x<myN_x+3; x++)
        {
            for(int y=3; y<myN_y+3; y++)
            {
                v_x_t[x][y]  =   v_x[x][y]+Delta_t*k3_vx[x][y];
                v_y_t[x][y]  =   v_y[x][y]+Delta_t*k3_vy[x][y];
                h_t[x][y]    =   h[x][y]+Delta_t*k3_h[x][y];
            }
        }

        exchange(v_x_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(v_y_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);
        exchange(h_t, myN_x, myN_y, leftNeighbor, rightNeighbor, bottomNeighbor,
                topNeighbor, Comm2D, &status, strideType);

        f_vx(v_x_t,v_y_t,h_t, k4_vx);
        f_vy(v_x_t,v_y_t,h_t, k4_vy);
        f_h(v_x_t,v_y_t,h_t, k4_h);

        for(int x=3; x<myN_x+3; x++)
        {
            for(int y=3; y<myN_y+3; y++)
            {
                v_x[x][y]    =   v_x[x][y]+Delta_t*(k1_vx[x][y]/6+k2_vx[x][y]/3
                                                  +k3_vx[x][y]/3+k4_vx[x][y]/6);
                v_y[x][y]    =   v_y[x][y]+Delta_t*(k1_vy[x][y]/6+k2_vy[x][y]/3
                                                  +k3_vy[x][y]/3+k4_vy[x][y]/6);
                h[x][y]      =   h[x][y]+Delta_t*(k1_h[x][y]/6+k2_h[x][y]/3
                                                  +k3_h[x][y]/3+k4_h[x][y]/6);
            }
        }
	//write to file
        time=time+Delta_t;
//	if(int_t%(int)(1/Delta_t)==0)
//	{
//           sprintf(myFileName, "assignment1_Process_%d_%d_%d.data",int_t, myCoords[X], myCoords[Y]);
//            file.open(myFileName, ios::out);
//            for(i=3; i<myN_x+3; i++)
//            {
//                for(j=3; j<myN_y+3; j++)
//                {
//                    file << h[i][j] << "\t";
//                }
//                file << endl;
//            }
//           file.close();
//	}
    }


    //write to file
    //sprintf(name, "assignment_1.csv.%d_%d_%d",int_t, myCoords[X], myCoords[Y]);
    //file.open(name, ios::out);
    //dump_f(file, h);
    //file.close();

    if(myID==0)
    {
        wtime   = MPI_Wtime() - wtime;  // Record the end time and calculate elapsed time
        cout << "Simulation took total=" << wtime << " seconds and "
             << wtime/k << " seconds per iteration with "
             << N_Procs << " processes" << endl;
    }

    MPI_Finalize();

    return 0;

}


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

void dump_f(fstream &file, double** arr)
{
    file << "x" << "," << "y" << "," << "h" << endl;

    for(int i=3; i<myN_x+3; i++)
    {
        for(int j=3; j<myN_y+3; j++)
        {
           file << x_min+(i-3)*Delta_x << ","
               << y_min+(j-3)*Delta_y << ","
               << arr[i][j]  << endl;
        }
    }

}

void f_vx(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    for(int x=3; x<myN_x+3; x++)
    {
        for(int y=3; y<myN_y+3; y++)
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

void f_vy(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    for(int x=3; x<myN_x+3; x++)
    {
        for(int y=3; y<myN_y+3; y++)
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

void f_h(double** v_x, double** v_y, double** h, double** f)
{
    // Sixth order finite stencil
    for(int x=3; x<myN_x+3; x++)
    {
        for(int y=3; y<myN_y+3; y++)
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
    for(int x = 0; x<myN_x+6; x++)
    {
        arr[x][2]=arr[x][end-4];
        arr[x][end-3]=arr[x][3];

        arr[x][1]=arr[x][end-5];
        arr[x][end-2]=arr[x][4];

        arr[x][0]=arr[x][end-6];
        arr[x][end-1]=arr[x][5];
    }

    end = N_x + 6;
    for(int y = 0; y<myN_y+6; y++)
    {
        arr[2][y]       =   arr[end-4][y];
        arr[end-3][y]   =   arr[3][y];

        arr[1][y]       =   arr[end-5][y];
        arr[end-2][y]   =   arr[4][y];

        arr[0][y]       =   arr[end-6][y];
        arr[end-1][y]   =   arr[5][y];
     }
}

// Message exchange function for each of the process to share necessary information with neighboring processes.
void exchange(double** phi, int myN_x, int myN_y, int leftNeighbor, int rightNeighbor, int bottomNeighbor, int topNeighbor, MPI_Comm Comm2D, MPI_Status* status, MPI_Datatype strideType)
{

    MPI_Sendrecv(&(phi[3][3]),     myN_y, MPI_DOUBLE, leftNeighbor,
           0, &(phi[myN_x+3][3]), myN_y, MPI_DOUBLE, rightNeighbor,  0, Comm2D, status);
    MPI_Sendrecv(&(phi[myN_x+2][3]), myN_y, MPI_DOUBLE, rightNeighbor,
           0, &(phi[2][3]),       myN_y, MPI_DOUBLE, leftNeighbor,   0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][3]),     1,     strideType, bottomNeighbor,
           0, &(phi[3][myN_y+3]), 1,     strideType, topNeighbor,    0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][myN_y+2]), 1,     strideType, topNeighbor,
           0, &(phi[3][2]),       1,     strideType, bottomNeighbor, 0, Comm2D, status);

    MPI_Sendrecv(&(phi[4][3]),     myN_y, MPI_DOUBLE, leftNeighbor,
           0, &(phi[myN_x+4][3]), myN_y, MPI_DOUBLE, rightNeighbor,  0, Comm2D, status);
    MPI_Sendrecv(&(phi[myN_x+1][3]), myN_y, MPI_DOUBLE, rightNeighbor,
           0, &(phi[1][3]),       myN_y, MPI_DOUBLE, leftNeighbor,   0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][4]),     1,     strideType, bottomNeighbor,
           0, &(phi[3][myN_y+4]), 1,     strideType, topNeighbor,    0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][myN_y+1]), 1,     strideType, topNeighbor,
           0, &(phi[3][1]),       1,     strideType, bottomNeighbor, 0, Comm2D, status);

    MPI_Sendrecv(&(phi[5][3]),     myN_y, MPI_DOUBLE, leftNeighbor,
           0, &(phi[myN_x+5][3]), myN_y, MPI_DOUBLE, rightNeighbor,  0, Comm2D, status);
    MPI_Sendrecv(&(phi[myN_x][3]), myN_y, MPI_DOUBLE, rightNeighbor,
           0, &(phi[0][3]),       myN_y, MPI_DOUBLE, leftNeighbor,   0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][5]),     1,     strideType, bottomNeighbor,
           0, &(phi[3][myN_y+5]), 1,     strideType, topNeighbor,    0, Comm2D, status);
    MPI_Sendrecv(&(phi[3][myN_y]), 1,     strideType, topNeighbor,
           0, &(phi[3][0]),       1,     strideType, bottomNeighbor, 0, Comm2D, status);

}
