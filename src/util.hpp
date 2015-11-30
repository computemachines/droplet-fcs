#ifndef UTIL_H
#define UTIL_H

#include <string>

const std::string readfile(const std::string& filename);


void savearray(unsigned long *data, int n, const char* filename);

#endif
