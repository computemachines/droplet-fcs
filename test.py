#! /usr/bin/python

import matplotlib.pyplot as plt
import pickle, pickletools
import numpy as np
import fcs

out = fcs.fcs()

if len(out[0]) > 0:
    plt.hist(out[0], bins=500)
#plt.show()

debug = pickle.loads(out[2])
