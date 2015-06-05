
#include "Python.h"

#include <tuple>
#include "numpy/arrayobject.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "fcs.cpp"

FCS fcs;

extern "C" {
  
  static PyObject * fcs_fcs(PyObject *self, PyObject *args){
    int total, groupsize, totalDroplets, dropletsPerGroup;
    float endTime, photonsPerIntensityPerTime;
    
    PyArg_ParseTuple(args, "iiiiff", &total, &groupsize, &totalDroplets, &dropletsPerGroup,
		                     &endTime, &photonsPerIntensityPerTime);

    fcs.init();
    tuple<uint*, uint, long> results = fcs.run(total, groupsize, totalDroplets,
					       dropletsPerGroup, endTime,
					       photonsPerIntensityPerTime);
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
    Py_InitModule("fcs", FCSMethods);
    import_array();
  }
}
