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
  void init();
  tuple<uint*, uint, long> run(int totalDroplets,
			       int dropletsPerGroup,
			       float endtime, float photonsPerIntensityPerTime);
};

#endif
