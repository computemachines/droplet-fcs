#ifndef METASIMULATION
#define METASIMULATION

#include <string>

#include "CL/cl.hpp"

const std::string readFile(std::string);

class Simulation{
  bool initialized = false;
protected:
  cl::Platform platform;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  cl::Kernel kernel;
public:
  void init(const std::string source);
};


#endif
