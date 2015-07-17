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

void Simulation::init(string source, int rngReserved){
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

  string options = "-D RNGRESERVED="+to_string(rngReserved); //+" -D LOCALPHOTONSLEN="+to_string(localPhotonsLen);
  #ifdef DEBUG
  options += " -D DEBUG";
  #endif
  cout << "program.cl build options: " << options << endl;
  err = program.build(devices, options.c_str());
  
  string log;
  program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
#ifndef DEBUG
  if(err != CL_SUCCESS){
#endif
    cout << log << endl;
    usleep(2000);
#ifndef DEBUG
  }
#endif
  
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
