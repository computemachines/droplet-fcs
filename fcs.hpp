#ifndef SIMULATION
#define SIMULATION

#include <vector>
#include <tuple>


#include "CL/cl.hpp"

#include "simulation.hpp"

using namespace std;

// Executes a single fcs simulation with parameters
class FCS : public Simulation{
  cl::Buffer timestamps;
public:
  void init(int rngReserved=10000);
  tuple<uint*, uint, long, float*> run(uint totalDroplets=1,
			       uint workgroups=1,
			       uint workitems=1,
			       float endTime=10.0,
			       float photonsPerIntensityPerTime=1.0,
			       uint globalBufferSizePerWorkgroup=1000,
			       uint localBufferSizePerWorkitem=1000);
};

#endif
