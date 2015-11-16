#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>

#include "CL/cl.hpp"

class Simulation{
  bool initialized = false;
public:
  cl::Platform platform;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  cl::Kernel kernel;
public:
  void init(std::string source, std::string options);
};


#endif
