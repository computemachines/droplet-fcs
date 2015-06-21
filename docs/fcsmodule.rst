FCS Python API
==============

.. py:module:: fcs

.. py:data:: NO_PROFILING
.. py:data:: WITH_PROFILING
.. py:data:: PROFILING_ONLY

.. py:function:: fcs(totalDroplets, dropletsPerGroup, time, photonsPerIntensityPerTime, ) -> photons

   :param int totalDroplets: Total number of droplets in simulation
   :param int dropletsPerGroup: Number of droplets assigned to each OpenCL work group (must divide totalDroplets)
   :param float time: Simulation time in seconds
   :param float photonsPerIntensityPerTime: Photon density at single molecule maximum intensity
   :param int profiling: Either :py:data:`fcs.NO_PROFILING`, :py:data:`fcs.WITH_PROFILING` or :py:data:`fcs.WITH_PROFILING`.

   :return: photons, profilingData or both
   :rtype: :py:class:`numpy.ndarray`, :py:class:`fcs.ProfilingData`, or tuple(:py:class:`numpy.ndarray`, :py:class:`fcs.ProfilingData`)

   Returns photon times from gpu using physical and simulation parameters.

