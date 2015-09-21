#! /usr/bin/python

from fcs import fcs
import matplotlib.pyplot as plt

def plottimes(groupsize):
    times = []
    N = []
    print groupsize
    for n in range(groupsize, 10000, groupsize):
        times.append(fcs(n, groupsize)[1]*1e-6)
        N.append(n)
    return plt.plot(N, times)

plottimes(64)
plt.ylabel('[ms]')
plt.show()
