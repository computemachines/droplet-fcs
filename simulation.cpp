#include "simulation.hpp"

#include <iostream>
#include <fstream>
#include <cassert>
#include <unistd.h>

#include "CL/cl.hpp"
#include "CL/opencl.h"

using namespace std;

void Simulation::init(string source, string options){
  if(initialized)
    return;
  initialized = true;

  cl_int err;
  std::vector<cl::Device> devices;

  platform = cl::Platform::get();
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  context = cl::Context(vector<cl::Device>(1,devices[0]), NULL, NULL, NULL, &err);
  assert(err == CL_SUCCESS);
  
  queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

  cl::Program::Sources sources;

  sources.push_back(make_pair(source.c_str(), source.length()+1));
  program = cl::Program(context, sources, &err);
  assert(err == CL_SUCCESS);

  err = program.build(devices, options.c_str());
  
  string log;
  program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);

  if(err != CL_SUCCESS){
    printf("log (length: %d): %s\n", (int)log.size(), log.c_str());
    cout.flush();
  }
  
  assert(err == CL_SUCCESS);
  
  vector<cl::Kernel> kernels;
  err = program.createKernels(&kernels);
  kernel = kernels[0];
  assert(err == CL_SUCCESS);
};
