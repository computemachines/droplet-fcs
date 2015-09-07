#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <tuple>
#include <cassert>

#include "CL/cl.hpp"
#include "CL/opencl.h"

using namespace std;

const string readFile(const string& filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}

int main(){
  cl_int err;
  cl::Platform p = cl::Platform::get();
  vector<cl::Device> devices;
  p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cl::Context context = cl::Context(devices, NULL, NULL, NULL, &err);
  assert(err == CL_SUCCESS);

  cl::CommandQueue queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
  cl::Program::Sources sources;

  string source = readFile("test/res/atomic_dec_test.cl");
  sources.push_back(make_pair(source.c_str(), source.length()+1));
  cl::Program program = cl::Program(context, sources, &err);
  assert(err == CL_SUCCESS);

  err = program.build(devices, "-w");

  string log;
  program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
  if(err != CL_SUCCESS){
    printf("%s\n", log.c_str());
  }
  assert(err == CL_SUCCESS);
  
  vector<cl::Kernel> kernels;
  err = program.createKernels(&kernels);
  assert(err == CL_SUCCESS);
  cl::Kernel kernel = kernels[0];

  cl_int initial_countdown = 10;
  cl::Buffer countdown_buffer = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &initial_countdown, &err);
  assert(err == CL_SUCCESS);

  int num_workitems = 3;
  int num_workgroups = 2;

  int results_size = sizeof(cl_int)*num_workitems*num_workgroups;
  
  cl::Buffer results_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, \
    results_size, NULL, &err);
  assert(err == CL_SUCCESS);

  kernel.setArg(0, countdown_buffer);
  kernel.setArg(1, results_buffer);

  queue.enqueueNDRangeKernel(kernel, cl::NDRange(0),
			     cl::NDRange(num_workgroups*num_workitems),
			     cl::NDRange(num_workitems), NULL);
  cl_int results[num_workgroups][num_workitems];
  //(cl_int *)malloc(results_size);
  queue.enqueueReadBuffer(results_buffer, CL_TRUE, 0, results_size, results[0]);
  for(int n = 0; n < num_workgroups; n ++){
    for(int m = 0; m < num_workitems; m ++){
      printf("%d, ", results[n][m]);
    }
    printf("\n");
  }
  return 0;
}
