// program.cl
// All functions prefixed by MWC were taken from:
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
// I should probably get permission
//
// My code begins on line 218


#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LOCK(a) atom_cmpxchg(a, 0, 1) 
#define UNLOCK(a) atom_xchg(a, 0)

typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

// import _random_.cl

float timestep(float sigma){
  return pown(sigma, 2) / (6*DIFFUSIVITY);
}

float sigma(float timestep){
  return sqrt(6*DIFFUSIVITY*timestep);
}

float max_sigma(float3 position){
  float3 box = (float3)(.1f, .1f, .5f);
  return length(fmax(fabs(position)-box/2, (float3)(0.0f))) + 8e-4f;
}

float detectionIntensity(float3 position){ //detection volume 200nm x 200nm x 2.7um
  return exp(-dot(position*position, (float3)(2500, 2500, 13.717f))/2.0f);
}

void wrap(float3 *position){ // +- 1 maps to +- 10um
  *position = fmod((*position)+(float3)(.5f), (float3)(1))-(float3)(.5f);
}


// import _pickle_.cl

__kernel void kernel_func(__global uint* dropletsRemaining,
			  __global ulong* globalBuffer, //write only (thinking about mapping to host mem)
			  __global uint* globalMutex,
			  __local ulong* localBuffer,
			  __local uint* localMutex,
			  __global uint* photons
#ifdef DEBUG
			  , __global char* debug
#endif
			  ){
#define PICKLE_INITIAL_START debug
  int n = get_group_id(0);
  int m = get_local_id(0);
  UNLOCK(localMutex);
  UNLOCK(globalMutex);
  // on my integrated gpu globalBuffer usually isn't clean
  if(LOCK(localMutex) == 0){ // run once per workgroup
    for(int i = 0; i < GLOBALSIZE; i++){
      globalBuffer[i + n*GLOBALSIZE] = 0;
    }
    UNLOCK(localMutex);
  } // workitems seem to wait on siblings before completing if branch
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  
#ifdef DEBUG
  pkl_t pkl_droplet = pkl_init(debug + (2*get_global_id(0))*PICKLE_SIZE);
  pkl_open(&pkl_droplet); // LIST
  pkl_t pkl_photons = pkl_init(debug + (2*get_global_id(0)+1)*PICKLE_SIZE);
  pkl_open(&pkl_photons);
#endif

  // deterministic random number generator
  mwc64x_state_t rng; 
  MWC64X_SeedStreams(&rng, 100340, RNGRESERVED);

  __local uint localPhotonsPos;

  __private uint dropletId;
  while((dropletId = atomic_dec(dropletsRemaining))>0){
    float3 position = (float3)(0); //nextUfloat3(&rng);
    float intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
    float T_j = 0, dT_j = timestep(max_sigma(position));
    float CDFI_j = 0;
    float photon_i = 0, CDFphoton_i = -log(nextUfloat(&rng));
#ifdef DEBUG
    pkl_open(&pkl_droplet); // DICT
#endif
    do{
      // if photon(i) generated beyond current step(j) then step droplet(j)
      if(CDFphoton_i > CDFI_j){
	T_j += dT_j;
	CDFI_j += intensity*dT_j;

	dT_j = timestep(max_sigma(position));
	position += sigma(dT_j)*nextGfloat3(&rng);
	intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
	wrap(&position);
      }
      // if photon(i) generated is before next step(j) then compute arrival time
      if(CDFphoton_i < CDFI_j + intensity*dT_j){ 
	photon_i = (CDFphoton_i - CDFI_j)/intensity + T_j;
	while(LOCK(globalMutex)); // wait until get lock
	globalBuffer[localPhotonsPos] = (ulong)(photon_i*1e9);
	localPhotonsPos ++;
	UNLOCK(globalMutex);
	CDFphoton_i -= log(nextUfloat(&rng));
      }else{ // photon generated is not during this step
	CDFI_j = CDFI_j + intensity*dT_j;
      } // if(CDFphoton_i < CDFI_j + intensity*dT_j)
      
    }while(photon_i < ENDTIME && T_j < ENDTIME);
#ifdef DEBUG
    pkl_close(&pkl_droplet, DICT);
#endif 
  } // while(atomic_dec(dropletsRemaining)>0) 

  barrier(CLK_GLOBAL_MEM_FENCE);
  if(LOCK(globalMutex)==0)
    globalBuffer[0] = localPhotonsPos;
  
#ifdef DEBUG
  pkl_close(&pkl_droplet, LIST);
  pkl_close(&pkl_photons, LIST);
  pkl_end(&pkl_droplet);
  pkl_end(&pkl_photons);
#endif
}
