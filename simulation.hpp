#ifndef METASIMULATION
#define METASIMULATION

#include <string>

#include "CL/cl.hpp"

class Simulation{
  bool initialized = false;
protected:
  cl::Platform platform;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  cl::Kernel kernel;
public:
  void init(std::string source, int rngReserved=10000, int localPhotonsLen=1000);
};


#endif
