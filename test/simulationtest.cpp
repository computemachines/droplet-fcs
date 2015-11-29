#include "gtest/gtest.h"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "simulation.hpp"
#include "fcs.hpp"
#include "util.hpp"

// const std::string readFile(const std::string& filename){
//   std::ifstream sourcefile(filename);
//   const std::string source((std::istreambuf_iterator<char>(sourcefile)),
// 			   std::istreambuf_iterator<char>());

//   return source;
// }

namespace {
  class SimulationTest : public ::testing::Test {
  protected:
    Simulation simulation;
    SimulationTest(){
      simulation.init(readfile("./test/res/identity_null.cl"), "");
    }
  };
  // TEST_F(SimulationTest, KernelExists) {
  //   ASSERT_NE(simulation.kernel, NULL);
  // }
}
