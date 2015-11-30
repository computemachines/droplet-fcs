

#include <iostream>
#include <tuple>

#include "gtest/gtest.h"

#include "single.hpp"
#include "util.hpp"

namespace {
  class SingleTest : public ::testing::Test {
  protected:
    Single single;
  protected:
  };
  TEST_F(SingleTest, NonZeroPhotonsGenerated) {
    unsigned long *photons = single.run();
    savearray(photons, 1000000, "./fixed_dye_2015112902.times");
    for(int i = 0; i < 1000000; i ++){
      // std::printf("%d, %d \n", i, photons[i]);
      EXPECT_LT(0, photons[i]);
    }
  }
  TEST_F(SingleTest, PhotonsMonotonicTimes) {
    unsigned long *photons = single.run();
    for(int i = 0; i < 1000000-1; i ++){
      EXPECT_LT(photons[i], photons[i+1]);
    }
  }
}
	
