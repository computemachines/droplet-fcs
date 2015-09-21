
CXX = g++

CXXFLAGS = -g -fPIC -std=c++11 -I./src -I./include -I/usr/include/python3.4 -I/usr/lib/python3.4/dist-packages/numpy/core/include/numpy
LDFLAGS = -L./libs
LDLIBS = -lOpenCL

CXXFLAGS_TEST = $(CXXFLAGS)
LDFLAGS_TEST = $(LDFLAGS) -pthread
LDLIBS_TEST = $(LDLIBS) -lgtest

./src%.cpp./src/%.o:
	$(CXX) $(CXXFLAGS) -c $<

./test%.cpp./test/%.o:
	$(CXX) $(CXXFLAGS_TEST) -c $<

# ----------------
./test/run_tests: ./test/run_tests.o ./test/kerneltest.o ./src/fcs.o ./src/simulation.o
	$(CXX) $(LDFLAGS_TEST) ./test/run_tests.o ./test/kerneltest.o ./src/fcs.o ./src/simulation.o $(LDLIBS_TEST) -o ./test/run_tests

./build/fcs.so: ./src/fcsmodule.o ./src/fcs.o ./src/simulation.o
	$(CXX) -shared ./src/fcsmodule.o ./src/fcs.o ./src/simulation.o $(LDFLAGS) $(LDLIBS) -o ./build/fcs.so
# ----------------


test: ./test/run_tests
	DISPLAY=:0 ./test/run_tests

build: ./build/fcs.so

clean:
	rm -v ./src/*.o
	rm -v ./build/*.*o
	rm -v ./test/*.o
	rm -v ./test/run_tests

all: build test
