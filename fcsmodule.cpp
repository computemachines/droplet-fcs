#include "Python.h"

#include <tuple>
#include "numpy/arrayobject.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "fcs.cpp"

#define NO_PROFILING 0
#define WITH_PROFILING 1
#define PROFILING_ONLY 2

extern "C" {
  
  static PyObject * fcs_fcs(PyObject *self, PyObject *args){
    FCS fcs;
    int totalDroplets, dropletsPerGroup, profilingOption, maxPhotons;
    int rngReserved, localPhotonsLen;
    float endTime, photonsPerIntensityPerTime;
    
    // PyArg_ParseTuple(args, "iiffii", &totalDroplets, &dropletsPerGroup,
    // 		     &endTime, &photonsPerIntensityPerTime, &profilingOption,
    // 		     &maxPhotons, &rngReserved, &localPhotonsLen);

    

    tuple<ulong*, uint, long, char*, uint> results = fcs.run();
    ulong* data = get<0>(results);
    long time = get<2>(results);
    ulong bufferSize = get<1>(results);
    ulong numPhotons = data[0];
    assert(numPhotons < bufferSize);
    npy_intp photonsShape = {numPhotons};
    PyObject * numpy = PyArray_SimpleNewFromData(1, &photonsShape, NPY_ULONG, data+1);
    int debug_length = (int)get<4>(results);
    char * debug = get<3>(results);
    #ifdef DEBUG
    printf("debug out(%d): {", get<3>(results));
    for(int i = 0; i < debug_length; i++)
      printf("%d ", debug[i]);
    printf("}\n");
    PyObject * ret = Py_BuildValue("(Ols#)", numpy, time, get<3>(results), (int)get<4>(results));
    #else
    PyObject * ret = Py_BuildValue("(Ol)", numpy, time);
    #endif
    Py_INCREF(ret);
    return ret;
  }

  static PyMethodDef FCSMethods[] = {
    {"fcs", fcs_fcs, METH_VARARGS, "DOCSTRING blah"},
    {NULL, NULL, 0, NULL}
  };

  PyMODINIT_FUNC initfcs(void){
    PyObject *m = Py_InitModule("fcs", FCSMethods);
    PyModule_AddIntConstant(m, "NO_PROFILING", NO_PROFILING);
    PyModule_AddIntConstant(m, "WITH_PROFILING", WITH_PROFILING);
    PyModule_AddIntConstant(m, "PROFILING_ONLY", PROFILING_ONLY);
    import_array();
  }
}
