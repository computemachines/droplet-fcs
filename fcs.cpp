#include "fcs.hpp"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <tuple>
#include <time.h>

#include "CL/cl.hpp"
#include "Python.h"

#include "simulation.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

const string readFile(const string& filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}


using namespace std;

int main(int argc, char** argv){
  FCS fcs;
  fcs.init();
  std::tuple<uint*, uint, long> results = fcs.run();
  uint *data = get<0>(results);
  printf("results (length: %d) {", get<1>(results));
  for(int i = 0; i < get<1>(results); i++)
    printf("%d, ", data[i]);
  printf("}\n");
}


void FCS::init(int rngReserved){
  Simulation::init(readFile("program.cl"), rngReserved);
}

// metaBuffer is in global mem but owned by workgroup
// buffer is in local mem but owned by workitem
tuple<uint*, uint, long> FCS::run(uint totalDroplets,
				  uint workgroups,
				  uint workitems,
				  float endTime,
				  float photonsPerIntensityPerTime,
				  uint globalBufferSizePerWorkgroup,
				  uint localBufferSizePerWorkitem){
  #ifdef DEBUG
  printf("FCS#run()\n");
  #endif

  cl::Event kernelEvent;
  cl_int err;
  cl::Buffer globalBuffer = 
    cl::Buffer(context, CL_MEM_READ_WRITE,
	       workgroups*globalBufferSizePerWorkgroup*sizeof(cl_uint),
	       NULL, &err);
  if(err != CL_SUCCESS)
    printf("buffer create fail\n");
  assert(err == CL_SUCCESS);
  cl::Buffer dropletsRemaining = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_uint),
					    &totalDroplets, &err);
  if(err != CL_SUCCESS)
    printf("buffer create fail\n");
  assert(err == CL_SUCCESS);
  cl_uint endTimeNS = (cl_uint)(endTime*1e9);
  kernel.setArg(0, dropletsRemaining);
  kernel.setArg(1, globalBuffer);
  kernel.setArg(2, cl::__local(workitems*localBufferSizePerWorkitem*sizeof(cl_uint)));

  #ifdef DEBUG
  printf("workgroups x workitems: %dx%d\n", workgroups, workitems);
  
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  #endif

  queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(workgroups*workitems),
			     cl::NDRange(workitems), NULL, &kernelEvent);
  kernelEvent.wait();

  #ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &stop);
  long astart = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  long aend = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  printf("CPU Start: %d, Stop: %d, Elapsed: %d\n", stop.tv_nsec, start.tv_nsec,
  	 stop.tv_nsec-start.tv_nsec);
  printf("GPU Start: %d, End: %d, Elapsed: %d\n", astart, aend, aend-astart);
  #endif

  uint buffer[workgroups*globalBufferSizePerWorkgroup];
  queue.enqueueReadBuffer(globalBuffer, CL_TRUE, 0,
			  workgroups*globalBufferSizePerWorkgroup*sizeof(cl_uint), buffer);
  queue.finish();

  #ifdef DEBUG
  return make_tuple((uint *)buffer, (uint)(workgroups*globalBufferSizePerWorkgroup), aend-astart);
  #else
  return make_tuple((uint *)buffer, (uint)(workgroups*globalBufferSizePerWorkgroup), (long)0);
  #endif
}
