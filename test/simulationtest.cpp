#include "gtest/gtest.h"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "simulation.hpp"
#include "fcs.hpp"

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
      simulation.init(readFile("./test/res/identity_null.cl"), "");
    }
  };
  TEST_F(SimulationTest, buffersExist) {

  }
}
