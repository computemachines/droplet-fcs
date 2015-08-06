#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define LOCK(a) atom_cmpxchg(a, 0, 1) 
#define UNLOCK(a) atom_xchg(a, 0) 


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

float2 nextGaussVec2(mwc64x_state_t *s){ 
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
  return (float3)(nextGaussVec2(s), nextGaussVec2(s).x);
}

/* #define DIFFUSIVITY 1.0 */
float timestep(float sigma){
  return pown(sigma, 2) / (6*DIFFUSIVITY);
}

float sigma(float timestep){
  return sqrt(6*DIFFUSIVITY*timestep);
}

float max_sigma(float3 position){
  return 1.0; //8e-4; // distance/5 to 1um x 1um x 5um box around detection volume + 8nm
}

float detectionIntensity(float3 position){ //detection volume 200nm x 200nm x 2.7um
  return exp(-dot(position*position, (float3)(2500, 2500, 13.717))/2);
}

void wrap(float3 *position){ // +- 1 maps to +- 10um
  *position = fmod((*position)+(float3)(.5), (float3)(1))-(float3)(.5);
}

/* #define RNGRESERVED 10000  */
/* #define LOCALSIZE 1000  */
/* #define GLOBALSIZE 1000  */
/* #define PHOTONSPERINTENSITYPERTIME 2.0  */
/* #define ENDTIME 10.0 */
/* #define DEBUG_SIZE 20 */

#define _post_to_debug_(D, C) D[__debug_pos__] = C, __debug_pos__ ++
/* #define _post_multibyte_to_debug(_D, C, T)	\ */
/*   *((T *)(&D[__debug_pos__])) = C, __debug_pos__ += sizeof(T) */
#define _post_multibyte_to_debug_(D, C, N) for (int i=0; i < N; i++) { (_post_to_debug_(D, C >> 8*i)); }
#define pkl_init(D) uint __debug_pos__ = 0
#define pkl_end(D) (_post_to_debug_(D, '.'))
#define _mark_(D) (_post_to_debug_(D, '('))
#define LIST 'l'
#define DICT 'd'
#define TUPLE 't'
#define pkl_open(D) _mark_(D)
#define pkl_close(D, T) (_post_to_debug_(D, T))
#define pkl_log_char(D, C) (_post_to_debug_(D, 'K'), _post_to_debug_(D, C))
#define pkl_log_int(D, I) _post_to_debug_(D, 'J'); _post_multibyte_to_debug_(D, I, 4)
#define pkl_log_float(D, F) _post_to_debug_(D, 'G'); _post_multibyte_to_debug_(debug, F, 4) ; _post_multibyte_to_debug_(D, 0, 4)
#define pkl_log_long(D, L) _post_to_debug_(D, '\x8a'); _post_to_debug_(D, 0x08); _post_multibyte_to_debug_(D, L, 8)
/* #define pkl_log_float(D, F) (_post_to_debug_(D, 'G'), _post_multibyte_to_debug_(D, F, float)) */

__kernel void hello(__global uint* dropletsRemaining,
		    __global ulong* globalBuffer, //write only (thinking about mapping to host mem)
		    __local ulong* localBuffer 
                    #ifdef DEBUG
		    , __global char* debug
		    #endif
		    ){
  int n = get_global_id(0);
  int m = get_local_id(0);
  __global int *globalMutex;
  __local int *localMutex;

  for(int i = 0; i < GLOBALSIZE; i++){
    globalBuffer[i] = 0;
  }
  
  #ifdef DEBUG

  uint fourbytes = 0x12345678;

  int test = -10;
  
  pkl_init(debug);
  pkl_open(debug);
  /* pkl_open(debug);  */
  /* pkl_log_char(debug, 77);  */
  /* pkl_close(debug, LIST);  */
  /* pkl_log_char(debug, 170); */
  /* pkl_log_int(debug, fourbytes); */
  /* pkl_log_int(debug, -12); */
  /* long eightbytes =              0x7123456789ABCDEF; */
  /* float frac = -1.05; */
  /* int exp_sign = as_int(frac) &  0xFF800000; */
  /* _post_to_debug_(debug, 'G'); // '\x8a' */
  /* /\* _post_to_debug_(debug, 0x08); *\/ */
  /* _post_multibyte_to_debug_(debug, exp_sign, 4); */
  /* _post_multibyte_to_debug_(debug, 0, 4); */



  #endif
  mwc64x_state_t rng; 
  MWC64X_SeedStreams(&rng, 100340, RNGRESERVED);

  __local uint index;

  #ifdef DEBUG
  pkl_log_int(debug, *dropletsRemaining);
  #endif

  while(atomic_dec(dropletsRemaining)>0){
    float3 position = (float3)(-1, 0, 0); //nextUfloat3(&rng);
    float intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
    float T_j = 0, dT_j = timestep(max_sigma(position));
    float CDFI_j = 0;
    float photon_i = 0, CDFphoton_i = -log(nextUfloat(&rng));
    #ifdef DEBUG
    uint steps = 0;
    pkl_log_int(debug, (int)(CDFphoton_i*1e6));
    pkl_open(debug);
    #endif
    do{
      if(CDFphoton_i > CDFI_j){
	// step t_j
	T_j += dT_j;
	CDFI_j += intensity*dT_j;
	
	dT_j = timestep(max_sigma(position));
	printf("%f, ",position.x);
	position += (float3)(2, 0, 0)/(300); // += sigma(dT_j)*nextGfloat3(&rng);
	intensity = PHOTONSPERINTENSITYPERTIME*detectionIntensity(position);
	wrap(&position); 
	#ifdef DEBUG
	pkl_log_int(debug, (int)(T_j*1e6));
	pkl_open(debug);
	pkl_open(debug);
	pkl_log_int(debug, (int)(position.x*1e6));
	pkl_log_int(debug, (int)(position.y*1e6));
	pkl_log_int(debug, (int)(position.z*1e6));
	pkl_close(debug, TUPLE);
	pkl_log_int(debug, (int)(intensity*1e6));
	pkl_close(debug, LIST);
	steps ++;
	#endif
      }

      if(CDFphoton_i < CDFI_j + intensity*dT_j){
	photon_i = (CDFphoton_i - CDFI_j)/intensity + T_j;
	globalBuffer[index] = (ulong)(photon_i*1e9);
	index ++;
	CDFphoton_i -= log(nextUfloat(&rng));
	/* #ifdef DEBUG */
	/* pkl_log_int(debug, index); */
	/* #endif */
      }else{
	CDFI_j = CDFI_j + intensity*dT_j;
	photon_i = T_j; // do not save to buffer. this is only to prevent endless do-while
      }
    }while(photon_i < ENDTIME);
    #ifdef DEBUG
    pkl_close(debug, DICT);
    pkl_log_int(debug, steps);
    #endif
  }

  //this is a hack. for some reason index=0 -> 0 droplets but
  // index=n -> (n-1) droplets for n>1
  if(index==0)
    index ++;
  globalBuffer[0] = index-1;
  
  #ifdef DEBUG
  pkl_close(debug, LIST);
  pkl_end(debug);
  #endif
}
