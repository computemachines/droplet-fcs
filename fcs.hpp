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
  tuple<uint*, uint, long> run(int totalDroplets,
			       int workgroups,
			       int workitems,
			       float endtime,
			       float photonsPerIntensityPerTime,
			       int globalPhotonMetaBufferCount,
			       int localPhotonBufferCount);
};

#endif
