#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LOCK(a) atom_cmpxchg(a, 0, 1) 
#define UNLOCK(a) atom_xchg(a, 0)

__kernel void atomic_dec_test(__global int *counter,
			      __global int *results){
  int myCount = 0;
  while(atomic_dec(counter)>0){
    myCount ++;
  }
  results[get_global_id(0)] = myCount;
}
