#ifndef SIMULATION
#define SIMULATION

#include <vector>

#include "CL/cl.hpp"

#include "simulation.cpp"

// Executes a single fcs simulation with parameters
class FCS : public Simulation{
  bool initialized = false;
  cl::Buffer timestamps;
public:
  void init();
  std::pair<float*, long> run(int total, int groupsize);
};

#endif
