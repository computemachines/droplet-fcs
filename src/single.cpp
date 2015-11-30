#include "single.hpp"

#include <cassert>

#include "CL/cl.hpp"

#include "util.hpp"

unsigned long* Single::run(){
  cl_int err;
  Simulation::init(readfile("res/_generated/single.cl"), " -w");
  unsigned long* photons = new unsigned long[1000000];
  cl::Buffer photonsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, 1000000*sizeof(cl_ulong), NULL, &err);
  assert(err == CL_SUCCESS);

  cl::Event kernelEvent;
  kernel.setArg(0, photonsBuffer);
  queue.enqueueTask(kernel, NULL, &kernelEvent);
  kernelEvent.wait();
  queue.enqueueReadBuffer(photonsBuffer, CL_TRUE, 0, 1000000*sizeof(cl_ulong), photons);
  queue.finish();
  return photons;
}
