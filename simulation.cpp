#include "simulation.hpp"

#include <iostream>
#include <fstream>
#include <cassert>
#include <unistd.h>

#include "CL/cl.hpp"
#include "CL/opencl.h"

using namespace std;

// int main(int argc, char** argv){

// #ifdef CURSES
//   CursesGUI().tick();
// #endif

//   Simulation simulations = Simulation::createForPlatform(readFile("program.cl"));
//   FCS simulation = simulations.createSimulation();
//   simulation.run();
// }

void Simulation::init(string source){
  if(initialized)
    return;
  initialized = true;
  #ifdef DEBUG
  printf("Simulation#init()\n");
  #endif
  cl_int err;
  std::vector<cl::Device> devices;

  platform = cl::Platform::get();
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  printf("Devices found: %d\n", (int)devices.size());

  cout << "Device Vendor ID: " << devices.front().getInfo<CL_DEVICE_VENDOR_ID>() << endl;

  context = cl::Context(vector<cl::Device>(1,devices[0]), NULL, NULL, NULL, &err);
  assert(err == CL_SUCCESS);
  
  queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

  cl::Program::Sources sources;

  sources.push_back(make_pair(source.c_str(), source.length()+1));
  program = cl::Program(context, sources, &err);
  assert(err == CL_SUCCESS);

  err = program.build(devices, "-DSINGLE");
  
  string log;
  program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
if(err != CL_SUCCESS){
    cout << log << endl;
usleep(1000);
}
  
  assert(err == CL_SUCCESS);
  
  vector<cl::Kernel> kernels;
  err = program.createKernels(&kernels);
  kernel = kernels[0];
  assert(err == CL_SUCCESS);
};


// cl::Buffer Simulation::createCorrBuff(size_t size){
//   //   return cl::Buffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);
//   
// }
