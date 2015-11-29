#include "util.hpp"

#include <iostream>
#include <fstream>
#include <cstdio>

using namespace std;

const string readfile(const string& filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}
