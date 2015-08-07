#include "Python.h"
#include <tuple>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "fcs.cpp"
#include "simulation.hpp"

extern "C" {
  static PyObject * fcs_fcs(PyObject *self, PyObject *args){
    physical_parameters physicalParameters;
    simulation_parameters simulationParameters;
    physicalParameters.totalDroplets = 1;
    physicalParameters.endTime = 1.0;
    physicalParameters.photonsPerIntensityPerTime = 1000.0;
    physicalParameters.diffusivity = 1.5;
    simulationParameters.workgroups = 1;
    simulationParameters.workitems = 1;
    simulationParameters.globalBufferSizePerWorkgroup = 100000;
    simulationParameters.localBufferSizePerWorkitem = 1000;
    simulationParameters.rngReserved = 1000;
    simulationParameters.debugSize = 1000;

    PyArg_ParseTuple(args, "|ifffiiiiii",
		     &physicalParameters.totalDroplets,
		     &physicalParameters.endTime,
		     &physicalParameters.photonsPerIntensityPerTime,
		     &physicalParameters.diffusivity,
		     &simulationParameters.workgroups,
		     &simulationParameters.workitems,
		     &simulationParameters.globalBufferSizePerWorkgroup,
		     &simulationParameters.localBufferSizePerWorkitem,
		     &simulationParameters.rngReserved,
		     &simulationParameters.debugSize);

    FCS fcs;
    FCS_out results = fcs.run(physicalParameters, simulationParameters);

    ulong *photons = get<0>(results);
    
    npy_intp photonsNumpyShape = {get<1>(results)};
    PyObject *photonsNumpy = PyArray_SimpleNewFromData(1, &photonsNumpyShape,
						       NPY_ULONG, photons);
    #ifndef DEBUG
    return photonsNumpy;
    #else

    long time = get<2>(results);

    int debugSize = get<4>(results);
    char *debug = get<3>(results);

    PyObject * ret = Py_BuildValue("(Ols#)", photonsNumpy, time,
				   debug, debugSize);
    Py_INCREF(ret);
    return ret;
    #endif
  }

  static PyMethodDef FCSMethods[] = {
    {"fcs", fcs_fcs, METH_VARARGS, "DOCSTRING blah"},
    {NULL, NULL, 0, NULL} // sentinel
  };

  PyMODINIT_FUNC initfcs(void){
    PyObject *m = Py_InitModule("fcs", FCSMethods);
    // PyModule_AddIntConstant(m, "NO_PROFILING", NO_PROFILING);
    // PyModule_AddIntConstant(m, "WITH_PROFILING", WITH_PROFILING);
    // PyModule_AddIntConstant(m, "PROFILING_ONLY", PROFILING_ONLY);
    import_array();
  }
}
