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
