#include <iostream>
// #include <fstream>
// #include <cstdio>
// #include <vector>
// #include <tuple>
// #include <cassert>

// #include "CL/cl.hpp"
// #include "CL/opencl.h"
#include "gtest/gtest.h"

//#include "kerneltest.cpp"

// namespace {
//   class KernelTest : public ::testing::Test {
//   public:
//     cl::Platform platform;
//     std::vector<cl::Device> devices;
//     cl::Context context;
//     cl::CommandQueue queue;
//     cl::Program::Sources sources;
//     std::string source;
//     cl::Program program;
//     std::vector<cl::Kernel> kernels;
//     cl::Kernel kernel;
//     cl_int initial_countdown = 10000;
//     int num_workitems = 10;
//     int num_workgroups = 10;
//     cl_int **results;
//   protected:
//     KernelTest(std::string testFileName) {
//       cl_int err;
//       platform = cl::Platform::get();
//       platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
//       context = cl::Context(devices, NULL, NULL, NULL, &err);
//       assert(err == CL_SUCCESS);

//       queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
//       source = readFile(testFileName);
//       sources.push_back(make_pair(source.c_str(), source.length()+1));
//       program = cl::Program(context, sources, &err);
//       assert(err == CL_SUCCESS);

//       err = program.build(devices, "-w");

//       string log;
//       program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
//       if(err != CL_SUCCESS){
// 	printf("%s\n", log.c_str());
//       }
//       assert(err == CL_SUCCESS);
  
//       err = program.createKernels(&kernels);
//       assert(err == CL_SUCCESS);
//       kernel = kernels[0];

//       cl::Buffer countdown_buffer = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &initial_countdown, &err);
//       assert(err == CL_SUCCESS);

//       int results_size = sizeof(cl_int)*num_workitems*num_workgroups;  
//       cl::Buffer results_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, \
// 					     results_size, NULL, &err);
//       assert(err == CL_SUCCESS);

//       kernel.setArg(0, countdown_buffer);
//       kernel.setArg(1, results_buffer);

//       queue.enqueueNDRangeKernel(kernel, cl::NDRange(0),
// 				 cl::NDRange(num_workgroups*num_workitems),
// 				 cl::NDRange(num_workitems), NULL);
//       *results = (cl_int *)malloc(num_workgroups*num_workitems*sizeof(cl_uint));
//       queue.enqueueReadBuffer(results_buffer, CL_TRUE, 0, results_size, results[0]);
//     }
//     void printResultsArray(cl_int results[][]) {
//       for(int n = 0; n < num_workgroups; n ++){
// 	for(int m = 0; m < num_workitems; m ++){
// 	  printf("%d, ", results[n][m]);
// 	}
// 	printf("\n");
//       }
//     }
    
//   };
//   TEST_F(KernelTest, Enqueue) {
//     EXPECT_EQ(100, results[0][0]);
//   }

// }// namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

