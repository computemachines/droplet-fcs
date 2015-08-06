#ifndef FCS_H
#define FCS_H

#include <vector>
#include <tuple>
#include <unordered_map>

#include "CL/cl.hpp"

#include "simulation.hpp"

#ifndef DEBUG
typedef std::tuple<ulong*, uint> FCS_out;
#else
typedef std::tuple<ulong*, uint, long, char*, uint> FCS_out;
#endif

// Ideally only the physical parameters should affect results
struct physical_parameters {
  uint totalDroplets;
  float endTime;
  float photonsPerIntensityPerTime;
  float diffusivity;
};

struct simulation_parameters {
  uint workgroups;
  uint workitems;
  uint globalBufferSizePerWorkgroup;
  uint localBufferSizePerWorkitem;
  // the number of unique random numbers allocated per generator
  uint rngReserved;

  uint debugSize;
};

// Executes a single fcs simulation with parameters
class FCS : public Simulation{
public:
  FCS_out run(physical_parameters physicalParameters,
	      simulation_parameters simulationParameters);
private:
  void init(std::string source, std::string options);
  std::string buildOptions(physical_parameters physicalParameters,
			   simulation_parameters simulationParameters);
};

#endif
