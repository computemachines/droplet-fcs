// fcsmodule.cpp

#include "Python.h" // must be included first
#include <tuple>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "fcs.hpp"

using namespace std;

extern "C" {
  static PyObject * fcs_fcs(PyObject *self, PyObject *args){
    // default parameters
    physical_parameters physicalParameters;
    simulation_parameters simulationParameters;
    debug_parameters debugParameters;
    physicalParameters.totalDroplets = 1;
    physicalParameters.endTime = 1.0;
    physicalParameters.photonsPerIntensityPerTime = 100000.0;
    physicalParameters.diffusivity = 1.5;
    simulationParameters.workgroups = 1;
    simulationParameters.workitems = 1;
    simulationParameters.globalBufferSizePerWorkgroup = 100000;
    simulationParameters.localBufferSizePerWorkitem = 1000;
    simulationParameters.rngReserved = 1000;
    debugParameters.debugSize = 1000000;
    debugParameters.pickleSize = 100000;

    PyArg_ParseTuple(args, "|ifffiiiiiii",
		     &physicalParameters.totalDroplets,
		     &physicalParameters.endTime,
		     &physicalParameters.photonsPerIntensityPerTime,
		     &physicalParameters.diffusivity,
		     &simulationParameters.workgroups,
		     &simulationParameters.workitems,
		     &simulationParameters.globalBufferSizePerWorkgroup,
		     &simulationParameters.localBufferSizePerWorkitem,
		     &simulationParameters.rngReserved,
		     &debugParameters.debugSize,
		     &debugParameters.pickleSize);

    FCS fcs;
    FCS_out results = fcs.run(physicalParameters,
			      simulationParameters
#ifdef DEBUG
			      ,debugParameters
#endif
			      );

    ulong *photons = get<0>(results);
    
    npy_intp photonsNumpyShape = {get<1>(results)};
    PyObject *photonsNumpy = PyArray_SimpleNewFromData(1, &photonsNumpyShape,
						       NPY_ULONG, photons);
    #ifndef DEBUG
    return photonsNumpy;
    #else

    long time = get<2>(results);

    vector<py_string> debug = get<3>(results);

    PyObject *ret = PyTuple_New(2+debug.size()); //photonsNumpy, time,
    PyTuple_SetItem(ret, 0, photonsNumpy);
    PyTuple_SetItem(ret, 1, Py_BuildValue("l", time));
    for(int i = 0; i < debug.size(); i++){
      py_string stringData = debug[i];
      PyObject *s = Py_BuildValue("s#", get<0>(stringData), get<1>(stringData));
      PyTuple_SetItem(ret, i+2, s);
    }
    return ret;
    #endif
  }

  static PyMethodDef FCSMethods[] = {
    {"fcs", fcs_fcs, METH_VARARGS, "DOCSTRING blah"},
    {NULL, NULL, 0, NULL} // sentinel
  };

  static struct PyModuleDef FCSModule = {
    PyModuleDef_HEAD_INIT,
    "fcs",
    NULL,
    -1,
    FCSMethods
  };

  PyObject* PyInit_fcs(){
    PyObject *m = PyModule_Create(&FCSModule);
    // PyModule_AddIntConstant(m, "NO_PROFILING", NO_PROFILING);
    // PyModule_AddIntConstant(m, "WITH_PROFILING", WITH_PROFILING);
    // PyModule_AddIntConstant(m, "PROFILING_ONLY", PROFILING_ONLY);
    import_array();
    return m;
  }
}
