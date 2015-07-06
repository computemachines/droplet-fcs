
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
    
    PyArg_ParseTuple(args, "iiffii", &totalDroplets, &dropletsPerGroup,
		     &endTime, &photonsPerIntensityPerTime, &profilingOption,
		     &maxPhotons, &rngReserved, &localPhotonsLen);

    

    fcs.init(rngReserved, localPhotonsLen);
    tuple<uint*, uint, long> results = fcs.run(totalDroplets,
					       dropletsPerGroup, endTime,
					       photonsPerIntensityPerTime, maxPhotons);
    uint* data = get<0>(results);
    long time = get<2>(results);
    npy_intp dims = {get<1>(results)};
    PyObject * numpy = PyArray_SimpleNewFromData(1, &dims, NPY_UINT, data);
    PyObject * ret = Py_BuildValue("(Ol)", numpy, time);
    // Py_INCREF(ret);
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
