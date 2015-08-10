// fcs.cpp
// Provides FCS which extends Simulation class. FCS specifies
// opencl program and preprocessor commands passed to
// Simulation::init. Photons arrival times and debugging info
// are passed back to caller.
//
// Called from fcsmodule.cpp

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

// This is supposed to silence a compiler warning, but I don't think
// it is working
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


using namespace std;

// TODO: move to utility library
const string readFile(const string& filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}

void FCS::init(std::string source, std::string options){
  Simulation::init(source, options);
}

std::string FCS::buildOptions(physical_parameters physicalParameters,
			      simulation_parameters simulationParameters){
  string options = 
    " -D ENDTIME="+to_string(physicalParameters.endTime) + "f"+
    " -D DIFFUSIVITY="+to_string(physicalParameters.diffusivity) + "f"+
    " -D RNGRESERVED="+to_string(simulationParameters.rngReserved) +
    " -D LOCALSIZE="+to_string(simulationParameters.localBufferSizePerWorkitem) +
    " -D GLOBALSIZE="+to_string(simulationParameters.globalBufferSizePerWorkgroup) +
    " -D PHOTONSPERINTENSITYPERTIME="+to_string(physicalParameters.photonsPerIntensityPerTime) +"f";

#ifdef DEBUG
  options += " -w -D DEBUG -D DEBUG_SIZE=" + to_string(simulationParameters.debugSize);
  #endif
  
  return options;
}

FCS_out FCS::run(physical_parameters physicalParameters,
		 simulation_parameters simulationParameters){
  cl::Event kernelEvent;
  cl_int err;

  FCS::init(readFile("program.cl"),
	    buildOptions(physicalParameters, simulationParameters));

  // allocate global memory on gpu
  size_t globalBufferSize = simulationParameters.workgroups*\
    simulationParameters.globalBufferSizePerWorkgroup*sizeof(cl_ulong);
  cl::Buffer globalBuffer = cl::Buffer(context, CL_MEM_READ_WRITE,
				       globalBufferSize, NULL, &err);
  assert(err == CL_SUCCESS);
  
  // this allocates and copies totalDroplets(cpu) -> dropletsRemaining(gpu)
  cl::Buffer dropletsRemaining = \
    cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	       sizeof(cl_uint), &physicalParameters.totalDroplets, &err);
  assert(err == CL_SUCCESS);
  
  #ifdef DEBUG
  // holds the pickled objects from gpu
  cl::Buffer debugBuffer = \
    cl::Buffer(context, CL_MEM_READ_WRITE, simulationParameters.debugSize,
	       NULL, &err);
  assert(err == CL_SUCCESS);	
  #endif

  // allocate local memory in each workgroup
  size_t localBufferSize = simulationParameters.workitems*\
    simulationParameters.localBufferSizePerWorkitem*sizeof(cl_ulong);
  cl::LocalSpaceArg localBuffer = cl::__local(localBufferSize);

  // associate kernel arguments with buffers
  kernel.setArg(0, dropletsRemaining);
  kernel.setArg(1, globalBuffer);
  kernel.setArg(2, localBuffer);
  #ifdef DEBUG
  kernel.setArg(3, debugBuffer);
  #endif

  // struct timespec start, stop;
  // clock_gettime(CLOCK_REALTIME, &start);

  queue.enqueueNDRangeKernel(kernel, cl::NDRange(0),
			     cl::NDRange(simulationParameters.workgroups*\
					 simulationParameters.workitems),
			     cl::NDRange(simulationParameters.workitems), NULL,
			     &kernelEvent);
  kernelEvent.wait();

  // get profiling times
  // clock_gettime(CLOCK_REALTIME, &stop);
  long kernelBegin = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  long kernelEnd = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  // printf("CPU Start: %d, Stop: %d, Elapsed: %d\n",
  // 	 stop.tv_nsec, start.tv_nsec, stop.tv_nsec-start.tv_nsec);

  // transfer global buffers from gpu to cpu mem
  ulong globalBufferNumLongs;
  queue.enqueueReadBuffer(globalBuffer, CL_TRUE, 0, sizeof(long),
			  &globalBufferNumLongs);
  ulong *buffer = (ulong *)malloc(globalBufferNumLongs*sizeof(cl_ulong));
  queue.enqueueReadBuffer(globalBuffer, CL_TRUE, sizeof(cl_long),
			  globalBufferNumLongs*sizeof(cl_ulong), buffer);
  
  #ifdef DEBUG
  char *debugString = (char *)malloc(simulationParameters.debugSize);
  queue.enqueueReadBuffer(debugBuffer, CL_TRUE, 0,
			  simulationParameters.debugSize, debugString);
  #endif

  // wait for any transfers to finish
  queue.finish();

  #ifndef DEBUG
  // 'buffer' is wrapped in a numpy array of longs
  return make_tuple(buffer, globalBufferNumLongs);
  #else

  // debugString is unpicked in python caller
  return make_tuple(buffer, (uint)globalBufferNumLongs,
		    kernelEnd-kernelBegin, debugString,
		    simulationParameters.debugSize);
  #endif
}
