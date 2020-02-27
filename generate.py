from spice_loader import *

from scipy.linalg import norm

import numpy.random as npr

def station_coords(station_name, et0, req, flattening):
    xyz = spice.spkezr(station_name, et0, 'ITRF93', 'NONE', 'EARTH')[0][0:3] * 1000.0
    return spice.recgeo(xyz, req, flattening)

def generate_ground_measurements(name, object_id, stations, time_arange_args,
                                 min_elevation = 5 * np.pi/180.0):
    loader = SpiceLoader(name)

    station_xyz = {}
    station_times = {}
    station_ranges = {}
    station_range_rates = {}
    station_elevations = {}
    for station in stations:
        station_xyz[station]    = spice.spkezr(station, time_arange_args[0], 'ITRF93', 'NONE', 'EARTH')[0][0:3]
        station_times[station]  = []
        station_ranges[station] = []
        station_range_rates[station] = []
        station_elevations[station] = []

    times = np.arange(*time_arange_args)
    

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')

    xs = []

    for station in stations:
        for et in times:
            moon_occult = spice.occult(str(object_id), 'POINT', ' ', '301', 'ELLIPSOID',
                                       'IAU_MOON', 'NONE', station, et)
            
            # compute observer-target state
            x0, lt0 = spice.spkcpo(str(object_id), et, station+'_TOPO', 'OBSERVER',
                                       'NONE', station_xyz[station], 'EARTH', 'ITRF93')
            xs.append(x0)

            unit = x0[0:3] / norm(x0[0:3])
            v_component = unit.dot(x0[3:6]).item()
            

            # Get azimuth and elevation
            r, lon, lat = spice.reclat(x0[0:3])
            if moon_occult >= 0:
                if lat >= min_elevation:
                    station_times[station].append(et)

                    r_noise = npr.randn(1).item() * 1.0
                    v_noise = npr.randn(1).item() * 0.001
                    
                    station_ranges[station].append(r * 1000.0 + r_noise)
                    station_range_rates[station].append(v_component * 1000.0 + v_noise)
                    station_elevations[station].append(lat)
                else:
                    pass
                    #print("range = {} m, azimuth = {}, elevation = {}".format(r, -lon * 180/np.pi, lat * 180/np.pi))
            else:
                pass
                #print("moon occulting")

    xs = np.vstack((xs)).T
    ax1.scatter(xs[0,:], xs[1,:], xs[2,:], s=1)
    ax1.scatter([0], [0], [0])

    ax2 = fig.add_subplot(212)

    for station in stations:
        ax2.scatter((np.array(station_times[station]) - times[0]) / 3600.0, np.array(station_elevations[station]) * 180/np.pi, s=2, label=station)
    ax2.grid()
    ax2.set_ylabel("elevation")
    ax2.set_xlabel("time (h)")
    #plt.show()

    return station_times, station_ranges, station_range_rates, station_elevations
