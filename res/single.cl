// single.cl
// produce photons for single saturated droplet fixed in center of beam

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LOCK(a) atom_cmpxchg(a, 0, 1) 
#define UNLOCK(a) atom_xchg(a, 0)

typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

// import res/_random_.cl

__kernel void kernel_func(__global ulong* photons){
  mwc64x_state_t rng; 
  MWC64X_SeedStreams(&rng, 100340, 10000);

  float3 position = (float3)(0);
  float intensity = 0.001;
  float CDFI_j = 0;
  float CDFphoton_i = -log(nextUfloat(&rng));
  photons[0] = CDFphoton_i/intensity;
  CDFphoton_i -= log(nextUfloat(&rng));
  for(int i=1; i < 1000000; i ++){
    photons[i] = CDFphoton_i/intensity;
    if(photons[i] <= photons[i-1]){
      photons[i] = photons[i-1]+1;
    }
    CDFphoton_i -= log(nextUfloat(&rng));
  }
  photons[0] = 1;
}
