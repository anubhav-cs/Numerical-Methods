#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void daxpy(__global double* x,
                    __global double* y,
                    __global double* a,
                    __global double* out,
                    __local  double* scratch)                        
{
   size_t globalID      = get_global_id(0);
   size_t localID       = get_local_id(0);
   size_t workGroupID   = get_group_id(0);
   size_t workGroupSize = get_local_size(0);
   size_t numWorkGroups = get_num_groups(0);

   scratch[localID] = (*a)*x[globalID] + y[globalID];
   
   barrier(CLK_LOCAL_MEM_FENCE);

   out[globalID] = scratch[localID];

  

}
