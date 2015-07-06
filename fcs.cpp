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
  printf("the final test\n");
  printf("results (length: %d) {", get<1>(results));
  for(int i = 0; i < get<1>(results); i++)
    printf("%d, ", data[i]);
  printf("}\n");
}


void FCS::init(int rngReserved, int localPhotonsLen){
  Simulation::init(readFile("program.cl"), rngReserved, localPhotonsLen);
}

tuple<uint*, uint, long> FCS::run(int totalDroplets,
				  int dropletsPerGroup,
				  float endTime,
				  float photonsPerIntensityPerTime,
				  int maxPhotons){
  assert(totalDroplets%dropletsPerGroup == 0);

  #ifdef DEBUG
  printf("FCS#run()\n");
  #endif

  cl::Event kernelEvent;
  cl_int err;
  cl::Buffer photonsBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY,
					maxPhotons*sizeof(cl_uint), NULL, &err);
  cl_uint numPhotons = 0;
  cl::Buffer numPhotonsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
					   sizeof(cl_uint), &numPhotons, &err);

  cl_uint endTimeNS = (cl_uint)(endTime*1e9);
  kernel.setArg(0, endTimeNS);
  kernel.setArg(1, dropletsPerGroup);
  kernel.setArg(2, photonsPerIntensityPerTime);
  kernel.setArg(3, numPhotonsBuffer);
  kernel.setArg(4, photonsBuffer);
  kernel.setArg(5, cl::__local(1000*sizeof(cl_uint)));
  assert(err == CL_SUCCESS);

  #ifdef DEBUG
  printf("Total kernels/group size: %d/%d\n", total, groupsize);
  #endif
  
  // struct timespec start, stop;
  // clock_gettime(CLOCK_REALTIME, &start);
  queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(totalDroplets),
			     cl::NDRange(dropletsPerGroup), NULL, &kernelEvent);
  kernelEvent.wait();
  // clock_gettime(CLOCK_REALTIME, &stop);
  long astart = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  long aend = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  #ifdef DEBUG
  // printf("CPU Start: %d, Stop: %d, Elapsed: %d\n", stop.tv_nsec, start.tv_nsec,
  // 	 stop.tv_nsec-start.tv_nsec);
  printf("GPU Start: %d, End: %d, Elapsed: %d\n", astart, aend, aend-astart);
  #endif

  queue.enqueueReadBuffer(numPhotonsBuffer, CL_TRUE, 0, sizeof(cl_uint), &numPhotons);
  queue.finish();
  cl_uint *photons = (cl_uint *)malloc(numPhotons*sizeof(cl_uint));
  for(int i=0; i < numPhotons; i++)
    photons[i] = 0;
  queue.enqueueReadBuffer(photonsBuffer, CL_TRUE, 0, numPhotons*sizeof(cl_uint), photons);
  queue.finish();

  return make_tuple(photons, numPhotons, aend-astart);
}
