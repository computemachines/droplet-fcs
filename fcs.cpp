#include "fcs.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>

#include "CL/cl.hpp"
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

const string readFile(string filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}



using namespace std;

int main(int argc, char** argv){
  FCS fcs;
  fcs.init();
  std::pair<float*, long> results = fcs.run(20, 5);
  float *data = results.first;
  
  printf("result {");
  for(int i = 0; i < 20; i++)
    printf("%5.3f, ", data[i]);
  printf("}\n");
}


void FCS::init(){
  Simulation::init(readFile("program.cl"));
}


std::pair<float*, long> FCS::run(int total, int groupsize){
  #ifdef DEBUG
  printf("FCS#run()\n");
  #endif

  cl::Event kernelEvent;
  cl_int err;
  cl::Buffer timestamps = cl::Buffer(context, CL_MEM_WRITE_ONLY, total*sizeof(int), NULL, &err);
  kernel.setArg(0, timestamps);
  assert(err == CL_SUCCESS);

  #ifdef DEBUG
  printf("Total kernels/group size: %d/%d\n", total, groupsize);
  #endif
  
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(total),
			     cl::NDRange(groupsize), NULL, &kernelEvent);
  kernelEvent.wait();
  clock_gettime(CLOCK_REALTIME, &stop);
  float *data = (float *)malloc(sizeof(float)*total);
  for(int i = 0; i < total; i++)
    data[i] = 0;
  long astart = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  long aend = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  #ifdef DEBUG
  printf("CPU Start: %d, Stop: %d, Elapsed: %d\n", stop.tv_nsec, start.tv_nsec,
	 stop.tv_nsec-start.tv_nsec);
  printf("GPU Start: %d, End: %d, Elapsed: %d\n", astart, aend, aend-astart);
  #endif
  queue.enqueueReadBuffer(timestamps, CL_TRUE, 0, total*sizeof(float), data);
  queue.finish();

  return make_pair(data, aend-astart);
}
