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

// Pre: a<M, b<M 
// Post: r=(a+b) mod M 
ulong MWC_AddMod64(ulong a, ulong b, ulong M) 
{
  ulong v=a+b; 
  if( (v>=M) || (v<a) )
    v=v-M; 
  return v; 
} 

// Pre: a<M,b<M 
// Post: r=(a*b) mod M 
// This could be done more efficently, but it is portable, and should 
// be easy to understand. It can be replaced with any of the better 
// modular multiplication algorithms (for example if you know you have 
// double precision available or something). 
ulong MWC_MulMod64(ulong a, ulong b, ulong M) 
{	 
  ulong r=0; 
  while(a!=0){ 
    if(a&1) 
      r=MWC_AddMod64(r,b,M); 
    b=MWC_AddMod64(b,b,M); 
    a=a>>1; 
  } 
  return r; 
} 


// Pre: a<M, e>=0 
// Post: r=(a^b) mod M 
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on 
// most architectures 
ulong MWC_PowMod64(ulong a, ulong e, ulong M) 
{ 
  ulong sqr=a, acc=1; 
  while(e!=0){ 
    if(e&1) 
      acc=MWC_MulMod64(acc,sqr,M); 
    sqr=MWC_MulMod64(sqr,sqr,M); 
    e=e>>1; 
  } 
  return acc; 
} 

uint2 MWC_SkipImpl_Mod64(uint2 curr, ulong A, ulong M, ulong distance) 
{ 
  ulong m=MWC_PowMod64(A, distance, M); 
  ulong x=curr.x*(ulong)A+curr.y; 
  x=MWC_MulMod64(x, m, M); 
  return (uint2)((uint)(x/A), (uint)(x%A)); 
} 

uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap) 
{ 
  // This is an arbitrary constant for starting LCG jumping from. I didn't 
  // want to start from 1, as then you end up with the two or three first values 
  // being a bit poor in ones - once you've decided that, one constant is as 
  // good as any another. There is no deep mathematical reason for it, I just 
  // generated a random number. 
  enum{ MWC_BASEID = 4077358422479273989UL }; 
	
  ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap; 
  ulong m=MWC_PowMod64(A, dist, M); 
	
  ulong x=MWC_MulMod64(MWC_BASEID, m, M); 
  return (uint2)((uint)(x/A), (uint)(x%A)); 
} 


//! Represents the state of a particular generator 
typedef struct{ uint4 x; uint4 c; } mwc64xvec4_state_t; 

enum{ MWC64XVEC4_A = 4294883355U }; 
enum{ MWC64XVEC4_M = 18446383549859758079UL }; 

void MWC64XVEC4_Step(mwc64xvec4_state_t *s) 
{ 
  uint4 X=s->x, C=s->c; 
	
  uint4 Xn=MWC64XVEC4_A*X+C; 
  // Note that vector comparisons return -1 for true, so we have to do this odd negation 
  // I would hope that the compiler would do something sensible if possible... 
  uint4 carry=as_uint4(-(Xn<C));		 
  uint4 Cn=mad_hi((uint4)MWC64XVEC4_A,X,carry); 
	
  s->x=Xn; 
  s->c=Cn; 
} 

void MWC64XVEC4_Skip(mwc64xvec4_state_t *s, ulong distance) 
{ 
  uint2 tmp0=MWC_SkipImpl_Mod64((uint2)(s->x.s0,s->c.s0), MWC64XVEC4_A, MWC64XVEC4_M, distance); 
  uint2 tmp1=MWC_SkipImpl_Mod64((uint2)(s->x.s1,s->c.s1), MWC64XVEC4_A, MWC64XVEC4_M, distance); 
  uint2 tmp2=MWC_SkipImpl_Mod64((uint2)(s->x.s2,s->c.s2), MWC64XVEC4_A, MWC64XVEC4_M, distance); 
  uint2 tmp3=MWC_SkipImpl_Mod64((uint2)(s->x.s3,s->c.s3), MWC64XVEC4_A, MWC64XVEC4_M, distance); 
  s->x=(uint4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x); 
  s->c=(uint4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y); 
} 

void MWC64XVEC4_SeedStreams(mwc64xvec4_state_t *s, ulong baseOffset, ulong perStreamOffset) 
{ 
  uint2 tmp0=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 0, baseOffset, perStreamOffset); 
  uint2 tmp1=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 1, baseOffset, perStreamOffset); 
  uint2 tmp2=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 2, baseOffset, perStreamOffset); 
  uint2 tmp3=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 3, baseOffset, perStreamOffset); 
  s->x=(uint4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x); 
  s->c=(uint4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y); 
} 

//! Return a 32-bit integer in the range [0..2^32) 
uint4 MWC64XVEC4_NextUint4(mwc64xvec4_state_t *s) 
{ 
  uint4 res=s->x ^ s->c; 
  MWC64XVEC4_Step(s); 
  return res; 
} 

typedef struct{ uint2 x; uint2 c; } mwc64xvec2_state_t; 

enum{ MWC64XVEC2_A = 4294883355U }; 
enum{ MWC64XVEC2_M = 18446383549859758079UL }; 

void MWC64XVEC2_Step(mwc64xvec2_state_t *s) 
{ 
  uint2 X=s->x, C=s->c; 
	
  uint2 Xn=MWC64XVEC2_A*X+C; 
  // Note that vector comparisons return -1 for true, so we have to do this negation 
  // I would hope that the compiler would do something sensible if possible... 
  uint2 carry=as_uint2(-(Xn<C));		 
  uint2 Cn=mad_hi((uint2)MWC64XVEC2_A,X,carry); 
	
  s->x=Xn; 
  s->c=Cn; 
} 

void MWC64XVEC2_Skip(mwc64xvec2_state_t *s, ulong distance) 
{ 
  uint2 tmp0=MWC_SkipImpl_Mod64((uint2)(s->x.s0,s->c.s0), MWC64XVEC2_A, MWC64XVEC2_M, distance); 
  uint2 tmp1=MWC_SkipImpl_Mod64((uint2)(s->x.s1,s->c.s1), MWC64XVEC2_A, MWC64XVEC2_M, distance); 
  s->x=(uint2)(tmp0.x, tmp1.x); 
  s->c=(uint2)(tmp0.y, tmp1.y); 
} 

void MWC64XVEC2_SeedStreams(mwc64xvec2_state_t *s, ulong baseOffset, ulong perStreamOffset) 
{ 
  uint2 tmp0=MWC_SeedImpl_Mod64(MWC64XVEC2_A, MWC64XVEC2_M, 2, 0, baseOffset, perStreamOffset); 
  uint2 tmp1=MWC_SeedImpl_Mod64(MWC64XVEC2_A, MWC64XVEC2_M, 2, 1, baseOffset, perStreamOffset); 
  s->x=(uint2)(tmp0.x, tmp1.x); 
  s->c=(uint2)(tmp0.y, tmp1.y); 
} 

//! Return a 32-bit integer in the range [0..2^32) 
uint2 MWC64XVEC2_NextUint2(mwc64xvec2_state_t *s) 
{ 
  uint2 res=s->x ^ s->c; 
  MWC64XVEC2_Step(s); 
  return res; 
} 


//! Represents the state of a particular generator 
typedef struct{ uint x; uint c; } mwc64x_state_t; 

enum{ MWC64X_A = 4294883355U }; 
enum{ MWC64X_M = 18446383549859758079UL }; 

void MWC64X_Step(mwc64x_state_t *s) 
{ 
  uint X=s->x, C=s->c; 
	
  uint Xn=MWC64X_A*X+C; 
  uint carry=(uint)(Xn<C);				// The (Xn<C) will be zero or one for scalar 
  uint Cn=mad_hi(MWC64X_A,X,carry);   
	
  s->x=Xn; 
  s->c=Cn; 
} 

void MWC64X_Skip(mwc64x_state_t *s, ulong distance) 
{ 
  uint2 tmp=MWC_SkipImpl_Mod64((uint2)(s->x,s->c), MWC64X_A, MWC64X_M, distance); 
  s->x=tmp.x; 
  s->c=tmp.y; 
} 

void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset) 
{ 
  uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset); 
  s->x=tmp.x; 
  s->c=tmp.y; 
} 

//! Return a 32-bit integer in the range [0..2^32) 
uint MWC64X_NextUint(mwc64x_state_t *s) 
{ 
  uint res=s->x ^ s->c; 
  MWC64X_Step(s); 
  return res; 
}

// Remaining is my work

float2 nextGfloat2(mwc64x_state_t *s){ 
  float u1 = MWC64X_NextUint(s)/pow(2.0f, 32); 
  float u2 = MWC64X_NextUint(s)/pow(2.0f, 32); 
  return (float2)(sqrt(-2*log(u1))*cospi(2*u2), 
 		  sqrt(-2*log(u1))*sinpi(2*u2));
}

float nextUfloat(mwc64x_state_t *s){
  return (float)(MWC64X_NextUint(s)/pow(2.0f, 32));
}

float3 nextUfloat3(mwc64x_state_t *s){
  return (float3)(MWC64X_NextUint(s)/pow(2.0f, 32),
		  MWC64X_NextUint(s)/pow(2.0f, 32),
		  MWC64X_NextUint(s)/pow(2.0f, 32));
}

float3 nextGfloat3(mwc64x_state_t *s){
  return (float3)(nextGfloat2(s), nextGfloat2(s).x);
}

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


uint64_t float_to_double(float val){
  //unsigned int v = as_uint(val);
  uint32_t v = *(uint32_t*) &val;
  uint64_t sign = (v >> 31) & 1;
  uint64_t exp = (v >> 23) & ((1 << 8) - 1);
  exp = exp - 127 + 1023;
  uint64_t sig = v & ((1 << 23) - 1);
  return (sign << 63) | (exp << 52) | (sig << (52-23));
}

// represents a pickle stream
typedef struct tag_pkl_t {
  __global char *head;
  int pos;
} pkl_t;

// pickle collection type enum
// these are the actual pickle opcodes
#define LIST 'l'
#define DICT 'd'
#define TUPLE 't'

// shallow copy of pkl_t (__global char *)
// pkl[0] is the length of the pickle string/stream
// use like:
//   pkl_t pkl = pkl_init(debug);
// then use &pkl for all other pkl_* calls
pkl_t pkl_init(__global char *debug){
  // reserve first 4 bytes for length of stream
  pkl_t pkl = {debug, 4};
  return pkl;
}

// private
// called from pickling functions to manage pointers
void _post_to_pkl_(pkl_t *pkl_ptr, char c){
  (pkl_ptr->head)[pkl_ptr->pos] = c;
  (pkl_ptr->pos) ++;
}


#define _post_multibyte_to_pkl_(D, C, N)	\
  for (int i=0; i < N; i++) {			\
    _post_to_pkl_(D, ((C >> 8*i) & 0xFF));	\
  }

#define _post_reversed_multibyte_to_pkl_(D, C, N)	\
  for (int i=N-1; i>=0 ; i--) {				\
    (_post_to_pkl_(D, ((C >> 8*i) & 0xFF)));		\
  }


// terminate a pickle stream
void pkl_end(pkl_t *pkl_ptr){
  _post_to_pkl_(pkl_ptr, '.');
  *(__global int *)(pkl_ptr->head) = pkl_ptr->pos - 4;
}

#ifdef PICKLE_SIZE
// check remaining chars in pickle stream before overflow into next pickle
int pkl_remaining(pkl_t *pkl_ptr){
  return PICKLE_SIZE - pkl_ptr->pos;
}
#endif

// write char to pickle stream
void pkl_log_char(pkl_t *pkl_ptr, char c){
  _post_to_pkl_(pkl_ptr, 'K');
  _post_to_pkl_(pkl_ptr, c);
}

// write mark opcode to pickle stream
void _mark_(pkl_t *pkl_ptr){
  _post_to_pkl_(pkl_ptr, '(');
}

// begin collection in pickle stream
// all following calls to pkl_log_* place objects in collection
// terminate collection with call to pkl_close
// example, writes {10: "hello worlds"} to the stream:
//   pkl_open(&pkl);
//   pkl_log_int(&pkl, 10);
//   pkl_log_string(&pkl, "hello worlds");
//   pkl_close(&pkl, DICT);
void pkl_open(pkl_t *pkl_ptr){
  _mark_(pkl_ptr);
}

// closes a collection and determines its type
// an unpaired call to pkl_open must precede a call to pkl_close
void pkl_close(pkl_t *pkl_ptr, char collection_type_enum){
  _post_to_pkl_(pkl_ptr, collection_type_enum);
}


void pkl_log_int(pkl_t *pkl_ptr, int n){ 
  _post_to_pkl_(pkl_ptr, 'J'); 
  _post_multibyte_to_pkl_(pkl_ptr, n, 4); 
} 

void _post_string_to_pkl_(pkl_t *pkl_ptr, __constant char *string){ 
  // reserve a char for the length of the string
  int str_len_pos = pkl_ptr->pos;
  (pkl_ptr->pos) ++;
  
  char c;
  char index; 
  for(index = 0; (c=string[index]) != 0; index++) 
    _post_to_pkl_(pkl_ptr, c);

  (pkl_ptr->head)[str_len_pos] = index;
} 

void pkl_log_long(pkl_t *pkl_ptr, long n){
  _post_to_pkl_(pkl_ptr, '\x8a');
  _post_to_pkl_(pkl_ptr, 8);
  _post_multibyte_to_pkl_(pkl_ptr, n, 8);
}

void pkl_log_float(pkl_t *pkl_ptr, float n){
  _post_to_pkl_(pkl_ptr, 'G');
  unsigned long stretched = float_to_double(n);
  _post_reversed_multibyte_to_pkl_(pkl_ptr, stretched, 8);
}

//#define pkl_log_str(D, S) _post_to_debug_(debug, 'U'); _post_string_to_debug_(_debug_(D), S)
void pkl_log_str(pkl_t *pkl_ptr, __constant char *s){
  _post_to_pkl_(pkl_ptr, 'U');
  _post_string_to_pkl_(pkl_ptr, s);
}

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
  // on my integrated gpu globalBuffer usually isn't clean
  /* if(LOCK(localMutex) == 0){ // run once per workgroup */
  /*   for(int i = 0; i < GLOBALSIZE; i++){  */
  /*     globalBuffer[i + n*GLOBALSIZE] = 0; */
  /*   } */
  /*   UNLOCK(localMutex); */
  /* } // workitems seem to wait on siblings before completing if branch */
  /* barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); */
  
  /* *(__global int *)debug = 10; */
  
#ifdef DEBUG
  pkl_t pkl_droplet = pkl_init(debug + 2*get_global_id(0)*PICKLE_SIZE);
  pkl_open(&pkl_droplet); // LIST
  pkl_t pkl_photons = pkl_init(debug + (2*get_global_id(0)+1)*PICKLE_SIZE);
  pkl_open(&pkl_photons);
  printf("%d\n", debug[0]);
#endif

  // deterministic random number generator
  mwc64x_state_t rng; 
  MWC64X_SeedStreams(&rng, 100340, RNGRESERVED);

  __local uint localPhotonsPos;

  while(atomic_dec(dropletsRemaining)>0){
    float3 position = (float3)(0); //nextUfloat3(&rng);
    float intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
    float T_j = 0, dT_j = timestep(max_sigma(position));
    float CDFI_j = 0;
    float photon_i = 0, CDFphoton_i = -log(nextUfloat(&rng));
#ifdef DEBUG
    pkl_open(&pkl_droplet); // DICT
#endif
    do{
      if(CDFphoton_i > CDFI_j){
	T_j += dT_j;
	CDFI_j += intensity*dT_j;
	
	dT_j = timestep(max_sigma(position));
	position += sigma(dT_j)*nextGfloat3(&rng);
	intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
	wrap(&position); 

#ifdef DEBUG
	/* if(pkl_remaining(&pkl_droplet) > 100){ */
	/*   pkl_log_float(&pkl_droplet, T_j); // key */
	/*   pkl_open(&pkl_droplet); // value // TUPLE */
	/*   pkl_log_float(&pkl_droplet, position.x);  */
	/*   pkl_log_float(&pkl_droplet, position.y);  */
	/*   pkl_log_float(&pkl_droplet, position.z);  */
	/*   pkl_close(&pkl_droplet, TUPLE); */
	/* } */
#endif
      } // if(CDFphoton_i > CDFI_j)
      
      if(CDFphoton_i < CDFI_j + intensity*dT_j){
	photon_i = (CDFphoton_i - CDFI_j)/intensity + T_j;
	/* globalBuffer[localPhotonsPos] = (ulong)(photon_i*1e9); */
	/* localPhotonsPos ++; */
	CDFphoton_i -= log(nextUfloat(&rng));
#ifdef DEBUG
	atom_inc(photons);
	//	pkl_log_float(&pkl_photons, photon_i);
#endif
      }else{
	CDFI_j = CDFI_j + intensity*dT_j;
	photon_i = T_j;
      } // if(CDFphoton_i < CDFI_j + intensity*dT_j)
    }while(photon_i < ENDTIME);
#ifdef DEBUG
    pkl_close(&pkl_droplet, DICT);
#endif
  } // while(atomic_dec(dropletsRemaining)>0)

  globalBuffer[0] = 0;
  
#ifdef DEBUG
  pkl_close(&pkl_droplet, LIST);
  pkl_end(&pkl_droplet);
  pkl_open(&pkl_photons);
  pkl_log_str(&pkl_photons, "total photons");
  pkl_log_int(&pkl_photons, *photons);
  pkl_close(&pkl_photons, DICT);
  pkl_close(&pkl_photons, LIST);
  pkl_end(&pkl_photons);
#endif

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  printf("%d.%d success\n", n, m);
}
