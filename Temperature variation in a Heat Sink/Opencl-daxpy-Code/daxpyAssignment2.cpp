/////////////////////////////////////////////////////////////////////////////
//
// Applied High Perfomance Computing
//
// Problem:		Computing daxpy with OpenCL
//
// Author :   Anubhav Singh
//
/////////////////////////////////////////////////////////////////////////////

#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

// Function declarations
void initialize(cl_platform_id& platformID, cl_device_id& deviceID, cl_context& context, cl_command_queue& commandQueue, cl_program& program, cl_kernel& daxpy, int D);
const char* read(const char* name);
double get_wtime();

int main(int argc, char** argv)
{
    unsigned int     N = 0;
    unsigned int     D = 0;
    cl_platform_id   platformID;
    cl_device_id     deviceID;
    cl_context       context;
    cl_command_queue commandQueue;
    cl_program       program;
    cl_kernel        daxpy;
    cl_char          deviceName[10240];
    cl_int           errorID;

    if(argc<3)
    {
        cout << "You need to input the size of the 1D array then the device to target, e.g. ./Tutorial_10 1000000 0" << endl;
        exit(1);
    }
    else
    {
        N = atoi(argv[1]);
        D = atoi(argv[2]);
    }

    initialize(platformID, deviceID, context, commandQueue, program, daxpy, D);

    double  wtime            = 0.0;
    cl_uint allocSize        = 0;
    cl_uint numIterations    = 1000;
    cl_uint maxComputeUnits  = 0;
    cl_uint numWorkGroups    = 0;
    size_t  workGroupSize    = 0;
    size_t  maxWorkGroupSize = 0;
    size_t  numWorkItems     = 0;

    clGetDeviceInfo(deviceID, CL_DEVICE_NAME,                sizeof(deviceName), &deviceName,       NULL);
    clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cl_uint),    &maxComputeUnits,  NULL);
    clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),     &maxWorkGroupSize, NULL);

    workGroupSize = maxWorkGroupSize/16;
    allocSize     = ((N-1)|(workGroupSize-1)) + 1;
    numWorkGroups = allocSize/workGroupSize;
    numWorkItems  = allocSize;

    std::cout << "N               = " << N                << std::endl
              << "allocSize       = " << allocSize        << std::endl
              << "numWorkItems    = " << numWorkItems     << std::endl
              << "maxComputeUnits = " << maxComputeUnits  << std::endl
              << "workGroupSize   = " << workGroupSize    << std::endl
              << "numWorkGroups   = " << numWorkGroups    << std::endl << std::flush;
    double a_c       = (double)rand()/RAND_MAX;// a in aX+Y = Y
    double* x_h      = new double [allocSize]; // x in aX+Y = Y
    double* y_h      = new double [allocSize]; // y in aX+Y = Y
    double* local_sum  = new double [N];       // localhost calculation storage

    // Initialize input vectors with some random numbers
    srand((unsigned int)time(0));
    for(int i=0; i<N; i++)
    {
        x_h[i] = (double)rand()/RAND_MAX;
    }
    for(int i=N; i<allocSize; i++)
    {
        x_h[i] = 0.0;
    }
    srand((unsigned int)time(0));
    for(int i=0; i<N; i++)
    {
        y_h[i] = (double)rand()/(RAND_MAX-2000000000);
    }
    for(int i=N; i<allocSize; i++)
    {
        y_h[i] = 0.0;
    }

    // Compute daxpy on the host a number of times
    wtime = get_wtime();
    for(cl_uint iter=0; iter<numIterations; iter++)
    {
      for(int i=0; i<N; i++)
      {
          local_sum[i] = a_c*x_h[i] + y_h[i];
      }
    }
    wtime = get_wtime() - wtime; // Record the end time and calculate elapsed time

    cout << "The daxpy computation took " << wtime << " seconds on the host" << endl;
    // for(int i=0; i<N; i++)
    // {
    //    cout << a_c <<", "<<  x_h[i] << ", " << y_h[i] << ", " <<  local_sum[i] << "," << endl;
    // }
    // cout << "" << endl ;
    // Compute daxpy on the device
    wtime = get_wtime();

    // Allocate memory on the device and copy across the input arrays a and b
    cl_mem x_d      = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, allocSize*sizeof(double),     x_h,  &errorID);
    cl_mem y_d      = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, allocSize*sizeof(double),     y_h,  &errorID);
    cl_mem a_d      = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(double), &a_c, &errorID);
    cl_mem out_d      = clCreateBuffer(context, CL_MEM_WRITE_ONLY ,                   allocSize*sizeof(double),     NULL,  &errorID);

    // Set the arguments of the kernel
    errorID  = clSetKernelArg(daxpy, 0, sizeof(cl_mem),                 &x_d);
    errorID |= clSetKernelArg(daxpy, 1, sizeof(cl_mem),                 &y_d);
    errorID |= clSetKernelArg(daxpy, 2, sizeof(cl_mem),                 &a_d);
    errorID |= clSetKernelArg(daxpy, 3, sizeof(cl_mem),               &out_d);
    errorID |= clSetKernelArg(daxpy, 4, workGroupSize*sizeof(double), NULL);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error setting kernel arguments" << endl;
        exit(1);
    }

    // Compute the daxpy on the device a number of times
    for(cl_uint iter=0; iter<numIterations; iter++)
    {
        // Execute kernel code
        errorID = clEnqueueNDRangeKernel(commandQueue, daxpy, 1, NULL, &numWorkItems, &workGroupSize, 0, NULL, NULL);
        if(errorID!=CL_SUCCESS)
        {
            cerr << "Error enqueueing kernel " << errorID << endl;
            exit(1);
        }
    }

    // Copy the output back to the host
    errorID = clEnqueueReadBuffer(commandQueue, out_d, CL_TRUE, 0, N*sizeof(double), y_h, 0, NULL, NULL);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error reading buffer " << errorID << endl;
        exit(1);
    }

    wtime = get_wtime() - wtime;

    cout << "The daxpy computation took " << wtime << " seconds on the " << deviceName << endl;

    // for(int i=0; i<N; i++)
    // {
    //    cout <<  y_h[i]  << endl;
    // }

	return 0;
}

void initialize(cl_platform_id& platformID, cl_device_id& deviceID, cl_context& context, cl_command_queue& commandQueue, cl_program& program, cl_kernel& daxpy, int D)
{
    cl_uint     N_Platforms;
    cl_uint     N_Devices;
    cl_uint     N_Kernels = 1;
    cl_int      errorID;
    const char* kernelSource[N_Kernels];

    // Find the OpenCL platforms
    clGetPlatformIDs(0, NULL, &N_Platforms);
    cl_platform_id platforms[N_Platforms];
    clGetPlatformIDs(N_Platforms, platforms, NULL);

    platformID = platforms[0];

    // Find the devices in the platform
    clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, 0, NULL, &N_Devices);
    cl_device_id devices[N_Devices];
    clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, N_Devices, devices, NULL);

    deviceID = D<N_Devices ? devices[D] : devices[N_Devices-1];

    // Create a compute context
    context = clCreateContext(0, 1, &deviceID, NULL, NULL, &errorID);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error creating context" << endl;
        exit(1);
    }

    // Create a command queue
    commandQueue = clCreateCommandQueue(context, deviceID, 0, &errorID);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error creating command queue" << endl;
        exit(1);
    }

    kernelSource[0] = read("daxpy.cl");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, N_Kernels, kernelSource, NULL, &errorID);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error creating program with source" << endl;
        exit(1);
    }

    // Build the program
    errorID = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errorID!=CL_SUCCESS)
    {
        cerr << "Error building program" << endl;
        size_t len;
        char   buffer[2048];
        clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        cerr << buffer << endl;
        exit(1);
    }

    // Create the compute kernel from the program
    daxpy = clCreateKernel(program, "daxpy", &errorID);
    if(errorID!=CL_SUCCESS)
    {
        cerr << "Error creating daxpy kernel" << endl;
        exit(1);
    }

    return;
}

const char* read(const char* name)
{
    ifstream file(name);
    string   sourceCode((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    size_t   length = sourceCode.length();
    char*    buffer = new char [length+1];
    sourceCode.copy(buffer, length);
    buffer[length] = '\0';

    return buffer;
}

double get_wtime()
{
   static int sec = -1;
   struct timeval tv;
   gettimeofday(&tv, NULL);
   if (sec < 0) sec = tv.tv_sec;
   return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}
