#include "gtest/gtest.h"

#include "CL/cl.hpp"

#include "simulation.hpp"
#include "fcs.hpp" // for readFile


namespace {
  class AllocateBufferTest : public ::testing::Test {
  public:
    Simulation simulation;
    int err;
    
    AllocateBufferTest(){
      simulation.init(readFile("./test/res/identity_null.cl"), "");
    }
  };

  TEST_F(AllocateBufferTest, GlobalRW) {
    cl::Buffer globalBuffer = cl::Buffer(simulation.context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
  }
  TEST_F(AllocateBufferTest, LocalRW) {
    cl::__local(sizeof(cl_int));
  }
  
    
}
