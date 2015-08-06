#ifndef FCS_H
#define FCS_H

#include <vector>
#include <tuple>


#include "CL/cl.hpp"

#include "simulation.hpp"

using namespace std;

// Executes a single fcs simulation with parameters
class FCS : public Simulation{
  cl::Buffer timestamps;
public:
  tuple<ulong*, uint, long, char*, uint> run(uint totalDroplets=1,
					      uint workgroups=1,
					      uint workitems=1,
					      float endTime=5.0,
					      float photonsPerIntensityPerTime=10000.0,
					      uint globalBufferSizePerWorkgroup=100000,
					      uint localBufferSizePerWorkitem=1000);
};

#endif
