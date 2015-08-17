#! /usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pickle import loads
from pickletools import dis
import numpy as np
import fcs

out = fcs.fcs(1)

if type(out) == np.ndarray: # compiled with scons debug=0 (default)
    pass # todo save to file or run through photon tools
elif type(out) == tuple: # compiled with scons debug=1
    if __name__ == "__main__":
        fig = plt.figure(figsize=(13.66, 7.68), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        debug = loads(out[2])
        trajectory = debug[0]
else:
    raise Exception("unexpected output")


def plot(end=len(trajectory), spin=False):
    X, Y, Z = zip(*[value for (key, value) in sorted(trajectory.items())])
    T = np.array(sorted(trajectory.keys()), dtype=float)

    # for i in range(1, end):
    #     ax.plot([X[i-1], X[i]], [Y[i-1], Y[i]], [Z[i-1], Z[i]], color='k')
    ax.plot(X, Y, Z, color='k')

    if spin:
        for theta in range(0, 360, 1):
            ax.view_init(20, theta)
            print "frame {}".format(theta)
            fig.savefig("test_imgs/frame_{}.png".format(theta))

#plot(alpha=lambda T: np.ones(T.shape), spin=True)
# if len(out[0]) > 0:
#     plt.hist(out[0], bins=500)
#plt.show()


