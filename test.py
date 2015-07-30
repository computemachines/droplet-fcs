#! /usr/bin/python

import matplotlib.pyplot as plt
import pickle
import fcs

out = fcs.fcs()

plt.hist(out[0], bins=100)
#plt.show()
