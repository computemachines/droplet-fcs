#include "util.hpp"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <tuple>

using namespace std;

const string readfile(const string& filename){
  ifstream sourcefile(filename);
  const string source((istreambuf_iterator<char>(sourcefile)),
		istreambuf_iterator<char>());

  return source;
}

void savearray(unsigned long *data, int n, const char* filename){
  FILE *file;
  file = fopen(filename, "wb");
  fwrite(data, sizeof(unsigned long), n, file);
  fclose(file);
}
