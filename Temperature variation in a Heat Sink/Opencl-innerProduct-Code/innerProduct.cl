#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void innerProduct(__global double* a,
                         __global double* aInner,
                         __local  double* scratch)
{   

   size_t globalID      = get_global_id(0);
   size_t localID       = get_local_id(0);
   size_t workGroupID   = get_group_id(0);
   size_t workGroupSize = get_local_size(0);
   size_t numWorkGroups = get_num_groups(0);

   scratch[localID] = a[globalID] * a[globalID];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(size_t i=workGroupSize>>1; i>0; i>>=1)
   {
      if(localID<i)
      {
         scratch[localID] += scratch[localID + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(localID==0)
   {
      aInner[workGroupID] = scratch[0];
   }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(globalID==0)
   {
       for(size_t i=1; i<numWorkGroups; i++)
       {
           aInner[0] += aInner[i];
       }
   }

}
