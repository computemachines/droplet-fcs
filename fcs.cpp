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
  std::tuple<uint*, uint, long> results = fcs.run(1, 1, 1.0, 1e-8);
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
tuple<uint*, uint, long> FCS::run(int totalDroplets,
				  int workgroups,
				  int workitems,
				  float endTime,
				  float photonsPerIntensityPerTime,
				  int globalBufferSizePerWorkgroup,
				  int localBufferSizePerWorkitem){
  #ifdef DEBUG
  printf("FCS#run()\n");
  #endif

  cl::Event kernelEvent;
  cl_int err;
  cl::Buffer globalBuffer = 
    cl::Buffer(context, CL_MEM_WRITE_ONLY,
	       workgroups*globalBufferSizePerWorkgroup*sizeof(cl_uint),
	       NULL, &err);
  cl::Buffer dropletsRemaining = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_uint),
					    &totalDroplets, &err);
  
  cl_uint endTimeNS = (cl_uint)(endTime*1e9);
  kernel.setArg(0, endTimeNS);
  kernel.setArg(1, dropletsRemaining);
  kernel.setArg(2, photonsPerIntensityPerTime);
  kernel.setArg(3, globalBuffer);
  kernel.setArg(4, cl::__local(workitems*localBufferSizePerWorkitem*sizeof(cl_uint))); //localPhotonBuffers
  assert(err == CL_SUCCESS);

  #ifdef DEBUG
  printf("workgroups x workitems: %d/x%d\n", workgroups, workitems);
  
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

  uint localBufferFlushes[workgroups];
  queue.enqueueReadBuffer(globalPhotonMetaBuffersPos, CL_TRUE, 0,
			  workgroups*sizeof(cl_uint), photonsMetaBuffersSize);
  uint photonsMetaBuffersPos
  for(int workgroup = 0; workgroup < workgroups; workgroup ++){
    
  }
  cl_uint *photons = (cl_uint *)malloc(numPhotons*sizeof(cl_uint));
  for(int i=0; i < numPhotons; i++)
    photons[i] = 0;
  queue.enqueueReadBuffer(globalPhotonMetaBuffers, CL_TRUE, 0, numPhotons*sizeof(cl_uint), photons);
  queue.finish();

  #ifdef DEBUG
  return make_tuple(photons, numPhotons, aend-astart);
  #else
  return make_tuple(photons, numPhotons, 0);
  #endif
}
