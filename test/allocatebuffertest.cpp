#include "gtest/gtest.h"

#include "CL/cl.hpp"


namespace {
  class AllocateBufferTest : public ::testing::Test {
  protected:

    AllocateBufferTest(){}
  };

  TEST_F(AllocateBufferTest, TestTest) {
    EXPECT_EQ(1, 1);
  }
}
