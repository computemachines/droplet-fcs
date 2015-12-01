# $@ target
# $< prereq 1

CXX = g++

CXXFLAGS = -g -fPIC -std=c++11 -I./src -I./include -I/usr/include/python3.4 -I/usr/lib/python3.4/dist-packages/numpy/core/include/numpy
LDFLAGS = -L./libs
LDLIBS = -lOpenCL

CXXFLAGS_TEST = $(CXXFLAGS)
LDFLAGS_TEST = $(LDFLAGS) -pthread
LDLIBS_TEST = $(LDLIBS) -lgtest

.PHONY: clean_target build_target test_target

all: build_target test_target
clean: clean_target

./src/%.o : ./src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

./test/%.o: ./test/%.cpp ./include/gtest/gtest.h
	$(CXX) $(CXXFLAGS_TEST) -c $< -o $@

./res/_generated/% : ./res/%
	./res/generate_resources.py $< -o $@

./libs:
	mkdir libs

./include:
	mkdir include

./include/gtest/gtest.h: ./include
	cp -vr ./googletest/googletest/include/gtest ./include/gtest

./googletest/googletest/make/gtest-all.o:
	cd ./googletest/googletest/make/; make all

./googletest/googletest/make/libgtest.a: ./googletest/googletest/make/gtest-all.o
	ar -rv ./googletest/googletest/make/libgtest.a ./googletest/googletest/make/gtest-all.o

./libs/libgtest.a: libs ./googletest/googletest/make/libgtest.a
	cp -v ./googletest/googletest/make/libgtest.a ./libs/libgtest.a

# ----------------
./test/run_tests: ./test/run_tests.o ./test/kerneltest.o ./test/simulationtest.o ./test/allocatebuffertest.o ./test/singletest.o ./src/fcs.o ./src/simulation.o ./src/single.o ./src/util.o ./libs/libgtest.a
	$(CXX) $(LDFLAGS_TEST) ./test/singletest.o ./test/run_tests.o ./test/kerneltest.o ./test/allocatebuffertest.o ./test/simulationtest.o ./src/fcs.o ./src/simulation.o ./src/single.o ./src/util.o $(LDLIBS_TEST) -o ./test/run_tests

./build:
	mkdir build

./build/fcs.so: ./src/fcsmodule.o ./src/fcs.o ./src/simulation.o ./src/util.o ./build
	$(CXX) -shared ./src/fcsmodule.o ./src/fcs.o ./src/simulation.o ./src/util.o $(LDFLAGS) $(LDLIBS) -o ./build/fcs.so
# ----------------


test_target: build_target ./test/run_tests
	DISPLAY=:0 ./test/run_tests

./res/_generated:
	mkdir ./res/_generated
resources: ./res/_generated ./res/_generated/single.cl ./res/_generated/program.cl 

build_target: ./build/fcs.so resources

clean_target:
	rm -v ./src/*.o
	rm -v ./build/*.*o
	rm -v ./test/*.o
	rm -v ./test/run_tests
	rm -v ./res/_generated/*
