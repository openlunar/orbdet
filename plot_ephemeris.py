from spice_loader import *
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ephemeris(spice_loader, fig = None, axes = None, object_id = -5440):
    et_begin, et_end = spice_loader.coverage(-5440)

    xs = []
    ms = []
    for et in np.arange(et_begin, et_end, 60.0):
        x = spice.spkezp( -5440, et, 'J2000', 'NONE', 399 )[0]
        m = spice.spkezp( 301,   et, 'J2000', 'NONE', 399 )[0]
        xs.append( x )
        ms.append( m )

    xs = np.vstack(xs).T * 1000.0
    ms = np.vstack(ms).T * 1000.0

    if fig is None and axes is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    axes.plot(xs[0,:], xs[1,:], xs[2,:], alpha=0.6)
    axes.plot(ms[0,:], ms[1,:], ms[2,:], alpha=0.6)
    axes.scatter([0], [0], [0])

    return fig, axes
    
