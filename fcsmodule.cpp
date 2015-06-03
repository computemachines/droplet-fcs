
#include "Python.h"

#include "numpy/arrayobject.h"

#include "fcs.cpp"

FCS fcs;

extern "C" {
  
  static PyObject * fcs_fcs(PyObject *self, PyObject *args){
    int num_droplets, group_size;
    PyArg_ParseTuple(args, "ii", &num_droplets, &group_size);

    fcs.init();
    std::pair<float*, long> results = fcs.run(num_droplets, group_size);
    float* data = results.first;
    long time = results.second;
    npy_intp dims = {num_droplets};
    PyObject * numpy = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, data);
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
