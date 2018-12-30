/////////////////////////////////////////////////////////////////////////////
//
// Applied Numerical Methods
//
// Problem:       rho.C.dT/dt = k. Grad^2 T
//
// Method:        Finite Element Method with 3D tetrahedral elements,
//                Implicit Euler Method and Conjugate Gradient Method
//
// Compilation:   g++ assignment2.cpp -o assignment
//
// Author :       Anubhav Singh
//
/////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <string.h>

using namespace std;

// Class definitions
class  SparseMatrix
{
public:
    SparseMatrix(int nrow, int nnzperrow)
    {
        // This constructor is called if we happen to know the number of rows
        // and an estimate of the number of nonzero entries per row.
        this->initialize(nrow, nnzperrow);
    }
    SparseMatrix()
    {
        // This constructor is called if we have no useful information
        N_row_      = 0;
        N_nz_      = 0;
        N_nz_rowmax_  = 0;
        N_allocated_  = 0;
        val_      = NULL;
        col_      = NULL;
        row_      = NULL;
        nnzs_      = NULL;
    }
   ~SparseMatrix()
    {
        if(val_)  delete  [] val_;
        if(col_)  delete  [] col_;
        if(row_)  delete  [] row_;
        if(nnzs_)  delete  [] nnzs_;
    }
    void  initialize(int nrow, int nnzperrow)
    {
        N_row_      = nrow;
        N_nz_      = 0;
        N_nz_rowmax_  = nnzperrow;
        N_allocated_  = N_row_*N_nz_rowmax_;
        val_      = new double  [N_allocated_];
        col_      = new int    [N_allocated_];
        row_      = new int    [N_row_+1];
        nnzs_      = new int    [N_row_+1];

        memset(val_,  0, N_allocated_ *sizeof(double));
        memset(row_,  0, (N_row_+1)   *sizeof(int));
        memset(nnzs_, 0, (N_row_+1)   *sizeof(int));

        for(int k=0; k<N_allocated_; k++)
        {
            col_[k] = -1;
        }

        for(int k=0, kk=0; k<N_row_; k++, kk+=N_nz_rowmax_)
        {
            row_[k] = kk;
        }

        return;
    }
    void  finalize()
    {
        int minCol    = 0;
        int insertPos  = 0;
        int  index    = 0;

        // Now that the matrix is assembled we can set N_nz_rowmax_ explicitly by
        // taking the largest value in the nnzs_ array
        N_nz_rowmax_ = 0;
        for(int m=0; m<N_row_; m++)
        {
            N_nz_rowmax_ = max(N_nz_rowmax_, nnzs_[m]);
        }

        double* tempVal    = new double [N_nz_];
        int*  tempCol    = new int   [N_nz_];
        int*  tempRow    = new int   [N_row_+1];
        bool*  isSorted  = new bool   [N_allocated_]; // This array will help us sort the column indices

        memset(tempVal,  0, N_nz_       *sizeof(double));
        memset(tempCol,  0, N_nz_       *sizeof(int));
        memset(tempRow,  0, (N_row_+1)  *sizeof(int));
        memset(isSorted, 0, N_allocated_*sizeof(bool));

        for(int m=0; m<N_row_; m++)
        {
            for(int k=row_[m]; k<(row_[m]+nnzs_[m]); k++)
            {
                minCol  = N_row_+1;
                for(int kk=row_[m]; kk<(row_[m]+nnzs_[m]); kk++)
                {
                    if(!isSorted[kk] && col_[kk]<minCol)
                    {
                        index    = kk;
                        minCol    = col_[index];
                    }
                }
                tempVal[insertPos]  = val_[index];
                tempCol[insertPos]  = col_[index];
                isSorted[index]    = true;
                insertPos++;
            }
            tempRow[m+1] = tempRow[m]+nnzs_[m];
        }

        delete [] val_;
        delete [] col_;
        delete [] row_;
        delete [] nnzs_;
        delete [] isSorted;

        val_    = tempVal;
        col_    = tempCol;
        row_    = tempRow;
        nnzs_    = NULL;
        N_allocated_  = N_nz_;

        return;
    }
    inline
    double& operator()(int m, int n)
    {
        // If the arrays are already full and inserting this entry would cause us to run off the end,
        // then we'll need to resize the arrays before inserting it
        if(nnzs_[m]>=N_nz_rowmax_)
        {
            this->reallocate();
        }
        // Search between row(m) and row(m+1) for col(k) = n (i.e. is the entry already in the matrix)
        int    k      = row_[m];
        bool  foundEntry  = false;
        while(k<(row_[m]+nnzs_[m]) && !foundEntry)
        {
            if(col_[k]==n)
            {
                foundEntry=true;
            }
            k++;
        }
        // If the entry is already in the matrix, then return a reference to it
        if(foundEntry)
        {
            return val_[k-1];
        }
        // If the entry is not already in the matrix then we'll need to insert it
        else
        {
            N_nz_++;
            nnzs_[m]++;
            col_[k]  = n;
            return val_[k];
        }
    }
    inline
    double&  operator()(int k)
    {
        return val_[k];
    }
    void  operator= (const SparseMatrix& A)
    {
        if(val_)  delete  [] val_;
        if(col_)  delete  [] col_;
        if(row_)  delete  [] row_;
        if(nnzs_)  delete  [] nnzs_;

        N_row_      = A.N_row_;
        N_nz_      = A.N_nz_;
        N_nz_rowmax_  = A.N_nz_rowmax_;
        N_allocated_  = A.N_allocated_;
        val_      = new double [N_allocated_];
        col_      = new int    [N_allocated_];
        row_      = new int    [N_row_+1];

        memcpy(val_, A.val_, N_nz_     *sizeof(double));
        memcpy(col_, A.col_, N_nz_     *sizeof(int));
        memcpy(row_, A.row_, (N_row_+1)*sizeof(int));
    }
    inline
    void  multiply(double* u, double* v)
    {
        // Note: This function will perform a matrix vector multiplication with the input vector v, returning the output in u.
        for(int m=0; m<N_row_; m++)
        {
            u[m] = 0.0;
            for(int k=row_[m]; k<row_[m+1]; k++)
            {
                u[m] += val_[k]*v[col_[k]];
            }
        }
        return;
    }
    inline
    void  multiply(double* u, double* v, bool* includerows, bool* includecols)
    {
        // Note: This function will perform a matrix vector multiplication on part of the matrix
        for(int m=0; m<N_row_; m++)
        {
            u[m] = 0.0;
            if(includerows[m])
            {
                for(int k=row_[m]; k<row_[m+1]; k++)
                {

                    if(includecols[col_[k]])
                    {
                        u[m] += val_[k]*v[col_[k]];
                    }
                }
            }
        }
        return;
    }
    inline
    void  subtract(double u, SparseMatrix& A)
    {
        for(int k=0; k<N_nz_; k++)
        {
            val_[k] -= (u*A.val_[k]);
        }
        return;
    }
    inline
    int    getNnz()
    {
        return N_nz_;
    }
    inline
    int    getNrow()
    {
        return N_row_;
    }
    void  print(const char* name)
    {
        fstream matrix;
        cout << "Matrix " << name << " has " << N_row_ << " rows with " << N_nz_ << " non-zero entries - " << N_allocated_ << " allocated." << flush;
        matrix.open(name, ios::out);
        matrix << "Mat = [" << endl;
        for(int m=0; m<N_row_; m++)
        {
            for(int n=row_[m]; n<row_[m+1]; n++)
            {
                matrix << m+1 << "\t" << col_[n]+1 << "\t" << val_[n] << endl;
            }
        }
        matrix << "];" << endl;
        matrix.close();
        cout << " Done." << flush << endl;
        return;
    }
protected:
    void  reallocate()
    {
        // Double the memory allocation size
        N_nz_rowmax_ *= 2;

        N_allocated_ = N_nz_rowmax_*N_row_;

        // Create some temporary arrays of the new size
        double* tempVal = new double [N_allocated_];
        int*  tempCol = new int    [N_allocated_];

        memset(tempVal, 0, N_allocated_*sizeof(double));
        memset(tempCol, 0, N_allocated_*sizeof(int));

        for(int m=0, mm=0; m<N_row_; m++, mm+=N_nz_rowmax_)
        {
            memcpy(&tempVal[mm], &val_[row_[m]], nnzs_[m]*sizeof(double));
            memcpy(&tempCol[mm], &col_[row_[m]], nnzs_[m]*sizeof(int));
            row_[m] = mm;
        }

        // Delete the memory used by the old arrays
        delete [] val_;
        delete [] col_;

        // Assign the addresses of the new arrays
        val_  = tempVal;
        col_  = tempCol;

        return;
    }
private:
    double*  val_;      // [N_nz]    Stores the nonzero elements of the matrix
    int*  col_;      // [N_nz]    Stores the column indices of the elements in each row
    int*  row_;      // [N_row+1] Stores the locations in val that start a row
    int*  nnzs_;      // [N_row+1] Stores the number of nonzero entries per row during the assembly process
    int    N_row_;      // The number of rows in the matrix
    int    N_nz_;      // The number of non-zero entries currently stored in the matrix
    int    N_nz_rowmax_;  // The maximum number of non-zero entries per row. This will be an estimate until the matrix is assembled
    int    N_allocated_;  // The number of non-zero entries currently allocated for in val_ and col_
};

class  Boundary
{
public:
    Boundary()
    {

    }
    string  name_;
    string  type_;
    int    N_;
    int*    indices_;
    double  value_;
};

// Global variables
const double  t_min       = 0.00;
const double  t_max       = 200.00;
const double    Delta_t     = 0.1;
const double  rho         = 8954.00;
const double  C           = 380.00;
const double    k           = 286.00;
const double    q_base      = 10000.00;
const double    h           = 100.00;
const double    T_air       = 300;
const int    N_t         = static_cast<int> ((t_max-t_min)/Delta_t+1);

// Function declarations
void  read(char* filename, double**& Points, int**& Faces, int**& Elements, Boundary*& Boundaries, int& N_p, int& N_f, int& N_e, int& N_b);
void  write(fstream& file, double* phi, int N_p);
void  write(double* phi, double**& Points, int**& Elements, int& myN_p, int& myN_e, int l);
void  assemble(SparseMatrix& M, SparseMatrix& K, double* s, double* phi, bool* Free, bool* Fixed, double** Points, int** Faces, int** Elements, Boundary* Boundaries, int N_p, int N_f, int N_e, int N_b);
void  solve(SparseMatrix& A, double* phi, double* b, bool* Free, bool* Fixed);
void    get_matrix_cofactor(int m, int n, int c, double matrix[][4], double co_mat[][4] ); // 4-D Matrix Cofactor
double  get_determinant(int dimension, double matrix[][4]) ; // 4-D Matrix Det

int     main(int argc, char** argv)
{
    // Simulation parameters
    double**        Points      = NULL;
    int**           Faces       = NULL;
    int**           Elements    = NULL;
    Boundary*       Boundaries  = NULL;
    int             N_p         = 0;
    int             N_f         = 0;
    int             N_e         = 0;
    int             N_b         = 0;
    double          t           = 0.0;
    fstream         file;

    if(argc<2)
    {
        cerr << "No grid file specified" << endl;
        exit(1);
    }
    else
    {
        read(argv[1], Points, Faces, Elements, Boundaries, N_p, N_f, N_e, N_b);
    }
    // Allocate arrays
    double*         phi         = new double [N_p];
    double*         s           = new double [N_p];
    double*         b           = new double [N_p];
    bool*           Free        = new bool   [N_p];
    bool*           Fixed       = new bool   [N_p];
    double*         AphiFixed  = new double [N_p];
    SparseMatrix    M;
    SparseMatrix    K;
    SparseMatrix    A;

    // Set initial condition
    t      = t_min;
    for(int m=0; m<N_p; m++)
    {
        phi[m]  = 300;
    }

    assemble(M, K, s, phi, Free, Fixed, Points, Faces, Elements, Boundaries, N_p, N_f, N_e, N_b);

    A = M;
    A.subtract(Delta_t, K); // At this point we have A = M-Delta_t*K

    // Compute the column vector to subtract from the right hand side to take account of fixed nodes
    A.multiply(AphiFixed, phi, Free, Fixed);

    // file.open("assignment2.data", ios::out);
    // write(file, phi, N_p);
    write(phi, Points, Elements, N_p, N_e, 0);

    // Time marching loop
    for(int l=0; l<N_t; l++)
    {
        t  += Delta_t;
        cout << "t = " << t;

        // Assemble b
        M.multiply(b, phi);
        for(int m=0; m<N_p; m++)
        {
            b[m]  += Delta_t*s[m] - AphiFixed[m];
        }

        // Solve the linear system
        solve(A, phi, b, Free, Fixed);

        // Write the solution
        if(l%(int)(1/Delta_t)==0)
    {
            write(phi, Points, Elements, N_p, N_e, l/(int)(1/Delta_t));
        }
    }

    file.close();

    // Deallocate arrays
    for(int bb=0; bb<N_b; bb++)
    {
        delete [] Boundaries[bb].indices_;
    }
    delete [] Points[0];
    delete [] Points;
    delete [] Faces[0];
    delete [] Faces;
    delete [] Elements[0];
    delete [] Elements;
    delete [] Boundaries;
    delete [] phi;
    delete [] s;
    delete [] b;
    delete [] Free;
    delete [] Fixed;
    delete [] AphiFixed;

    return 0;
}

void  read(char* filename, double**& Points, int**& Faces, int**& Elements, Boundary*& Boundaries, int& N_p, int& N_f, int& N_e, int& N_b)
{
    fstream    file;
    string      temp;

    cout << "Reading " << filename << "... " << flush;

    file.open(filename);
    if(!file.is_open())
    {
        cerr << "Error opening file" << endl;
        exit(1);
    }

    file >> temp >> N_p;
    file >> temp >> N_f;
    file >> temp >> N_e;
    file >> temp >> N_b;

    Points      = new double*  [N_p];
    Faces      = new int*    [N_f];
    Elements    = new int*    [N_e];
    Boundaries    = new Boundary  [N_b];
    Points[0]       = new double    [N_p*3];
    Faces[0]        = new int       [N_f*3];
    Elements[0]     = new int       [N_e*4];
    for(int p=1, pp=3; p<N_p; p++, pp+=3)
    {
        Points[p] = &Points[0][pp];
    }
    for(int f=1, ff=3; f<N_f; f++, ff+=3)
    {
        Faces[f] = &Faces[0][ff];
    }
    for(int e=1, ee=4; e<N_e; e++, ee+=4)
    {
        Elements[e] = &Elements[0][ee];
    }

    file >> temp;
    for(int p=0; p<N_p; p++)
    {
        file >> Points[p][0] >> Points[p][1] >> Points[p][2];;
    }

    file >> temp;
    for(int f=0; f<N_f; f++)
    {
        file >> Faces[f][0] >> Faces[f][1] >> Faces[f][2];
    }

    file >> temp;
    for(int e=0; e<N_e; e++)
    {
        file >> Elements[e][0] >> Elements[e][1] >> Elements[e][2] >> Elements[e][3];
    }

    file >> temp;
    for(int b=0; b<N_b; b++)
    {
        file >> Boundaries[b].name_ >> Boundaries[b].type_ >> Boundaries[b].N_;
        Boundaries[b].indices_  = new int [Boundaries[b].N_];
        for(int n=0; n<Boundaries[b].N_; n++)
        {
            file >> Boundaries[b].indices_[n];
        }
        file >> Boundaries[b].value_;
    }

    file.close();

    cout << "Done.\n" << flush;

    return;
}

void  write(fstream& file, double* phi, int N_p)
{
    for(int m=0; m<N_p; m++)
    {
        file << phi[m] << "\t";
    }
    file << "\n";
    return;
}

void  write(double* phi, double**& Points, int**& Elements, int& myN_p, int& myN_e, int l)
{
  fstream         file;
    char            fileName[64];

    sprintf(fileName, "VTKOutput/Assignment2_%03d.vtk", l);

    file.open(fileName, ios::out);

    file << "# vtk DataFile Version 2.0"<< endl;
    file << "untitled, Created by me"  << endl;
    file << "ASCII"            << endl;
    file << "DATASET UNSTRUCTURED_GRID"  << endl;

    file << "POINTS " << myN_p << " double" << endl;
    for(int p=0; p<myN_p; p++)
    {
        file << setw(6) << setprecision(5) << fixed << Points[p][0] << "\t" << Points[p][1] << "\t" << Points[p][2] << endl;
    }

    file << "CELLS " << myN_e << " " << 5*myN_e << endl;
    for(int e=0; e<myN_e; e++)
    {
        file << "4\t" << Elements[e][0] << "\t" << Elements[e][1] << "\t" << Elements[e][2] << "\t" << Elements[e][3] << endl;
    }

    file << "CELL_TYPES " << myN_e << endl;
    for(int e=0; e<myN_e; e++)
    {
        file << "10" << endl;
    }

    file << "POINT_DATA " << myN_p << endl;
    file << "SCALARS Temperature double 1" << endl;
    file << "LOOKUP_TABLE \"default\"" << endl;
    for(int p=0; p<myN_p; p++)
    {
        file << setprecision(5) << phi[p] << endl;
    }

    file.close();

  return;
}


void  assemble(SparseMatrix& M, SparseMatrix& K, double* s, double* phi, bool* Free, bool* Fixed, double** Points, int** Faces, int** Elements, Boundary* Boundaries, int N_p, int N_f, int N_e, int N_b)
{
    cout << "Assembling system... " << flush;

    double  x[4];
    double  y[4];
    double  z[4];
    double  gradEta[3][4];
    double  gradEta_p[3]= {0.0, 0.0, 0.0};
    double  gradEta_q[3]= {0.0, 0.0, 0.0};
    double  M_e[4][4]  = {{2.0, 1.0, 1.0, 1.0}, {1.0, 2.0, 1.0, 1.0}, {1.0, 1.0, 2.0, 1.0}, {1.0, 1.0, 1.0, 2.0}};
    double  R_o[3][3]  = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
    double  s_e[4]    = {1.0, 1.0, 1.0, 1.0};
    int    Nodes[4]  = {0, 0, 0, 0};
    double* Omega    = new double [N_e];
    double* Gamma    = new double [N_f];
    int    m;
    int    n;
    double  a = 0.00;
    double  b = 0.00;
    double  c = 0.00;

    // Assign all the indices to be free initially
    for(int p=0; p<N_p; p++)
    {
        Free[p] = 1;
        Fixed[p]= 0;
        s[p]    = 0.0;
    }

    // Calculate face areas
    for(int f=0; f<N_f; f++)
    {
        for(int p=0; p<3; p++)
        {
            x[p]  = Points[Faces[f][p]][0];
            y[p]  = Points[Faces[f][p]][1];
            z[p]  = Points[Faces[f][p]][2];
        }
        a = sqrt((x[1]-x[0])*(x[1]-x[0]) + (y[1]-y[0])*(y[1]-y[0]) + (z[1]-z[0])*(z[1]-z[0]));
        b = sqrt((x[2]-x[1])*(x[2]-x[1]) + (y[2]-y[1])*(y[2]-y[1]) + (z[2]-z[1])*(z[2]-z[1]));
        c = sqrt((x[2]-x[0])*(x[2]-x[0]) + (y[2]-y[0])*(y[2]-y[0]) + (z[2]-z[0])*(z[2]-z[0]));

        Gamma[f]  = sqrt((a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c))/4;
    }

    // Calculate element volumes
    for(int e=0; e<N_e; e++)
    {
        for(int p=0; p<4; p++)
        {
            x[p]          = Points[Elements[e][p]][0];
            y[p]          = Points[Elements[e][p]][1];
            z[p]          = Points[Elements[e][p]][2];
        }
        double matrix[4][4] =   {{1.0, x[0], y[0], z[0]},
                                {1.0, x[1], y[1], z[1]},
                                {1.0, x[2], y[2], z[2]},
                                {1.0, x[3], y[3], z[3]}
                                };

        Omega[e]            = abs(get_determinant(4, matrix))/6;
    }

    // Assemble M, K, and s
    M.initialize(N_p, 10);
    K.initialize(N_p, 10);

    for(int e=0; e<N_e; e++)
    {
        for(int p=0; p<4; p++)
        {
            Nodes[p]= Elements[e][p];
            x[p]  = Points[Nodes[p]][0];
            y[p]  = Points[Nodes[p]][1];
            z[p]  = Points[Nodes[p]][2];
        }
        // gradETA calculation
        gradEta[0][0] = (z[2-1]*(y[3-1]-y[4-1]) + z[3-1]*(y[4-1]-y[2-1]) + z[4-1]*(y[2-1]-y[3-1]))/(6*Omega[e]);
        gradEta[0][1] = (z[1-1]*(y[4-1]-y[3-1]) + z[3-1]*(y[1-1]-y[4-1]) + z[4-1]*(y[3-1]-y[1-1]))/(6*Omega[e]);
        gradEta[0][2] = (z[1-1]*(y[2-1]-y[4-1]) + z[2-1]*(y[4-1]-y[1-1]) + z[4-1]*(y[1-1]-y[2-1]))/(6*Omega[e]);
        gradEta[0][3] = (z[1-1]*(y[3-1]-y[2-1]) + z[2-1]*(y[1-1]-y[3-1]) + z[3-1]*(y[2-1]-y[1-1]))/(6*Omega[e]);

        gradEta[1][0] = (z[2-1]*(x[4-1]-x[3-1]) + z[3-1]*(x[2-1]-x[4-1]) + z[4-1]*(x[3-1]-x[2-1]))/(6*Omega[e]);
        gradEta[1][1] = (z[1-1]*(x[3-1]-x[4-1]) + z[3-1]*(x[4-1]-x[1-1]) + z[4-1]*(x[1-1]-x[3-1]))/(6*Omega[e]);
        gradEta[1][2] = (z[1-1]*(x[4-1]-x[2-1]) + z[2-1]*(x[1-1]-x[4-1]) + z[4-1]*(x[2-1]-x[1-1]))/(6*Omega[e]);
        gradEta[1][3] = (z[1-1]*(x[2-1]-x[3-1]) + z[2-1]*(x[3-1]-x[1-1]) + z[3-1]*(x[1-1]-x[2-1]))/(6*Omega[e]);

        gradEta[2][0] = (y[2-1]*(x[3-1]-x[4-1]) + y[3-1]*(x[4-1]-x[2-1]) + y[4-1]*(x[2-1]-x[3-1]))/(6*Omega[e]);
        gradEta[2][1] = (y[1-1]*(x[4-1]-x[3-1]) + y[3-1]*(x[1-1]-x[4-1]) + y[4-1]*(x[3-1]-x[1-1]))/(6*Omega[e]);
        gradEta[2][2] = (y[1-1]*(x[2-1]-x[4-1]) + y[2-1]*(x[4-1]-x[1-1]) + y[4-1]*(x[1-1]-x[2-1]))/(6*Omega[e]);
        gradEta[2][3] = (y[1-1]*(x[3-1]-x[2-1]) + y[2-1]*(x[1-1]-x[3-1]) + y[3-1]*(x[2-1]-x[1-1]))/(6*Omega[e]);

        // Outer loop over each node
        for(int p=0; p<4; p++)
        {
            m    = Nodes[p];
            gradEta_p[0]  = gradEta[0][p];
            gradEta_p[1]  = gradEta[1][p];
            gradEta_p[2]    = gradEta[2][p];

            // Inner loop over each node
            for(int q=0; q<4; q++)
            {
                n      = Nodes[q];
                gradEta_q[0]    = gradEta[0][q];
                gradEta_q[1]    = gradEta[1][q];
                gradEta_q[2]    = gradEta[2][q];

                M(m,n)     += rho*C*M_e[p][q]*Omega[e]/20;
                K(m,n)     -= (k*(gradEta_p[0]*gradEta_q[0]+gradEta_p[1]*gradEta_q[1]+gradEta_p[2]*gradEta_q[2])*Omega[e]);
            }
            s[m]       += 0;
        }
    }

    // Apply boundary conditions
    for(int b=0; b<N_b; b++)
    {
        if    (Boundaries[b].type_=="neumann")
        {
            for(int f=0; f<Boundaries[b].N_; f++)
            {
                for(int p=0; p<3; p++)
                {
                    Nodes[p]  = Faces[Boundaries[b].indices_[f]][p];
                    m      = Nodes[p];
                    s[m]     += q_base*Gamma[Boundaries[b].indices_[f]]/3;
                }
            }
        }
        else if  (Boundaries[b].type_=="robin")
        {
            for(int f=0; f<Boundaries[b].N_; f++)
            {
                for(int p=0; p<3; p++)
                {
                    Nodes[p]  = Faces[Boundaries[b].indices_[f]][p];
                    m      = Nodes[p];
                    s[m]     += h*T_air*Gamma[Boundaries[b].indices_[f]]/3;
                }

                for(int p=0; p<3; p++)
                {
                    m    = Nodes[p];
                    // Inner loop over each node
                    for(int q=0; q<3; q++)
                    {
                        n      = Nodes[q];
                        K(m,n)     -= (h*R_o[p][q]*Gamma[Boundaries[b].indices_[f]])/12;
                    }
                }
            }
        }
    }

    K.finalize();
    M.finalize();

    delete [] Gamma;
    delete [] Omega;

    cout << "Done.\n" << flush;

    return;
}

void  solve(SparseMatrix& A, double* phi, double* b, bool* Free, bool* Fixed)
{
    int    N_row      = A.getNrow();
    double*  r_old      = new double [N_row];
    double*  r        = new double [N_row];
    double*  d        = new double [N_row];
    double*  Ad        = new double [N_row];
    double*  Aphi      = new double [N_row];
    double  alpha      = 0.0;
    double  beta      = 0.0;
    double  r_norm      = 0.0;
    double  tolerance    = 1e-8;
    double  N_k             = 1e3;
    double  r_oldTr_old    = 0.0;
    double  rTr        = 0.0;
    double  dTAd      = 0.0;
    int    k        = 0;
    int    m        = 0;
    int    n        = 0;

    memset(r_old,    0, N_row*sizeof(double));
    memset(r,      0, N_row*sizeof(double));
    memset(d,      0, N_row*sizeof(double));
    memset(Ad,      0, N_row*sizeof(double));

    // Compute the initial residual
    A.multiply(Aphi, phi, Free, Free);
    for(m=0; m<N_row; m++)
    {
        if(Free[m])
        {
            r_old[m]  = b[m] - Aphi[m];
            d[m]    = r_old[m];
            r_oldTr_old+= r_old[m]*r_old[m];
        }
    }
    r_norm = sqrt(r_oldTr_old);

    // Conjugate Gradient iterative loop
    while(r_norm>tolerance && k<N_k)
    {
        dTAd  = 0.0;
        A.multiply(Ad, d, Free, Free);
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                dTAd   += d[m]*Ad[m];
            }
        }
        alpha    = r_oldTr_old/dTAd;
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                phi[m] += alpha*d[m];
            }
        }
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                r[m]  = r_old[m] - alpha*Ad[m];
            }
        }
        rTr  = 0.0;
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                rTr  += r[m]*r[m];
            }
        }
        beta    = rTr/r_oldTr_old;
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                d[m] = r[m] + beta*d[m];
            }
        }
        for(m=0; m<N_row; m++)
        {
            if(Free[m])
            {
                r_old[m] = r[m];
            }
        }
        r_oldTr_old  = rTr;
        r_norm    = sqrt(rTr);
        k++;
    }

    cout << ", k = " << k << ", r_norm = " << r_norm << endl;

    delete [] r_old;
    delete [] r;
    delete [] d;
    delete [] Ad;
    delete [] Aphi;

    return;
}

void get_matrix_cofactor(int m, int n, int c, double matrix[4][4], double co_mat[4][4] )
{
    int i = 0;
    int j = 0;
    int p = 0;
    for (i = 0; i < c; i++)
    {
        if (i != m )
        {
            int q = 0;
            for (j = 0; j < c; j++)
            {
                if (j != n)
                {
                    co_mat[p][q] = matrix[i][j];
                    q++;
                }
            }
            p++;
        }
    }
}

double get_determinant(int dimension, double matrix[][4])
{

    if (dimension == 1) return matrix[0][0];

    double determinant = 0.00;

    double new_temp[4][4];

    for (int i = 0; i < dimension; i++)
    {

        get_matrix_cofactor(0, i, dimension, matrix, new_temp);

        if (i%2==0){
            determinant += matrix[0][i] * get_determinant(dimension - 1, new_temp);
        }
        else{
            determinant -= matrix[0][i] * get_determinant(dimension - 1, new_temp);
        }

    }
    return determinant;
}
