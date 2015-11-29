

#include <iostream>
#include <tuple>

#include "gtest/gtest.h"

#include "single.hpp"

namespace {
  class SingleTest : public ::testing::Test {
  protected:
    Single single;
  protected:
  };
  TEST_F(SingleTest, NonZeroPhotonsGenerated) {
    unsigned long* photons = single.run();
    for(int i = 0; i < 10; i ++){
      std::printf("%d, %d \n", i, photons[i]);
      EXPECT_LT(0, photons[i]);
    }
  }
}
	
