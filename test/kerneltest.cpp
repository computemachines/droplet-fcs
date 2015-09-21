//#include "kerneltest.hpp"

#include <iostream>
#include <tuple>

#include "gtest/gtest.h"

#include "fcs.hpp"

namespace {
  class KernelTest : public ::testing::Test {
  protected:
     
    physical_parameters physicalParameters;
    simulation_parameters simulationParameters;
    debug_parameters debugParameters;
    FCS fcs;
    KernelTest() {
      physicalParameters.totalDroplets = 1;
      physicalParameters.endTime = 1.0;
      physicalParameters.photonsPerIntensityPerTime = 100000.0;
      physicalParameters.diffusivity = 1.5;
      simulationParameters.workgroups = 1;
      simulationParameters.workitems = 1;
      simulationParameters.globalBufferSizePerWorkgroup = 100000;
      simulationParameters.localBufferSizePerWorkitem = 1000;
      simulationParameters.rngReserved = 1000;
      debugParameters.debugSize = 1000000;
      debugParameters.pickleSize = 100000;
    }
    FCS_out run(){
      return fcs.run(physicalParameters,
		     simulationParameters
#ifdef DEBUG
		     ,debugParameters
#endif
		     );
    }
  };

  TEST_F(KernelTest, PositivePhotons) {
    FCS_out output = run();
    EXPECT_LE(0, std::get<0>(output)[0]);
  }
  TEST_F(KernelTest, PositivePhotonTimes) {
    FCS_out output = run();
    int num_photons = std::get<1>(output);
    ASSERT_LT(1, num_photons);
    ulong *photons = std::get<0>(output);
    for(int i = 0; i < num_photons-1; i++) {
      EXPECT_LE(photons[i], photons[i+1]);
    }
  }
}
