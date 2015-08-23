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
			      simulation_parameters simulationParameters
#ifdef DEBUG
			      ,debug_parameters debugParameters
#endif
			      ){
  string options = 
    " -D ENDTIME="+to_string(physicalParameters.endTime) + "f"+
    " -D DIFFUSIVITY="+to_string(physicalParameters.diffusivity) + "f"+
    " -D RNGRESERVED="+to_string(simulationParameters.rngReserved) +
    " -D LOCALSIZE="+to_string(simulationParameters.localBufferSizePerWorkitem) +
    " -D GLOBALSIZE="+to_string(simulationParameters.globalBufferSizePerWorkgroup) +
    " -D PHOTONSPERINTENSITYPERTIME="+to_string(physicalParameters.photonsPerIntensityPerTime) +"f";

#ifdef DEBUG
  options += " -w -D DEBUG";
  options += " -D DEBUG_SIZE=" + to_string(debugParameters.debugSize);
  options += " -D PICKLE_SIZE=" + to_string(debugParameters.pickleSize);
#endif
  
  return options;
}

FCS_out FCS::run(physical_parameters physicalParameters,
		 simulation_parameters simulationParameters
#ifdef DEBUG
		 ,debug_parameters debugParameters
#endif
		 ){
  cl::Event kernelEvent;
  cl_int err;

  FCS::init(readFile("program.cl"),
	    buildOptions(physicalParameters,
			 simulationParameters
#ifdef DEBUG
			 ,debugParameters
#endif
			 ));

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

  cl::Buffer globalMutex = cl::Buffer(context, CL_MEM_READ_WRITE,
				      sizeof(cl_uint), 0, &err);

  
  #ifdef DEBUG
  // holds the pickled objects from gpu
  cl::Buffer debugBuffer = \
    cl::Buffer(context, CL_MEM_READ_WRITE, debugParameters.debugSize,
	       NULL, &err);
  assert(err == CL_SUCCESS);	
  #endif

  // allocate local memory in each workgroup
  size_t localBufferSize = simulationParameters.workitems*\
    simulationParameters.localBufferSizePerWorkitem*sizeof(cl_ulong);
  cl::LocalSpaceArg localBuffer = cl::__local(localBufferSize);
  cl::LocalSpaceArg localMutex = cl::__local(sizeof(cl_uint));

  cl::Buffer numPhotons = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), 0, &err);
  // associate kernel arguments with buffers
  kernel.setArg(0, dropletsRemaining);
  kernel.setArg(1, globalBuffer);
  kernel.setArg(2, globalMutex);
  kernel.setArg(3, localBuffer);
  kernel.setArg(4, localMutex);
  kernel.setArg(5, numPhotons);
#ifdef DEBUG
  kernel.setArg(6, debugBuffer);
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
  
#ifndef DEBUG
  
  // wait for any transfers to finish
  queue.finish();

  // 'buffer' is wrapped in a numpy array of longs in caller fcsmodule.cpp
  return make_tuple(buffer, globalBufferNumLongs);

#else

  cl_int pickleLen = 1; // temporary positive value
  char *debugString;
  vector<py_string> pyStrings;
  
  for(int i = 0;
      i*debugParameters.pickleSize < debugParameters.debugSize; i ++){
    queue.enqueueReadBuffer(debugBuffer, CL_TRUE, i*debugParameters.pickleSize,
			    4, &pickleLen);
    queue.finish();
    if(pickleLen == 0)
      continue;
    debugString = (char *)malloc(pickleLen);
    queue.enqueueReadBuffer(debugBuffer, CL_TRUE,
			    4 + i*debugParameters.pickleSize,
			    pickleLen, debugString);
    pyStrings.push_back(make_tuple(debugString, pickleLen));
  }
  
  queue.finish();

  // debugString is unpicked in python caller
  return make_tuple(buffer, (uint)globalBufferNumLongs,
		    kernelEnd-kernelBegin, pyStrings);
#endif
}
