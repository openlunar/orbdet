from spice_loader import *

from scipy.linalg import norm

def station_coords(station_name, et0):
    xyz = spice.spkezr(station_name, et0, 'ITRF93', 'NONE', 'EARTH')[0][0:3] * 1000.0
    return spice.recgeo(xyz, req, flattening)

def generate_ground_measurements(name, object_id, stations, time_arange_args,
                                 min_elevation = 5 * np.pi/180.0):
    loader = SpiceLoader(name)

    station_xyz = {}
    station_times = {}
    station_ranges = {}
    station_range_rates = {}
    for station in stations:
        station_xyz[station]    = spice.spkezr(station, time_arange_args[0], 'ITRF93', 'NONE', 'EARTH')[0][0:3] * 1000.0
        station_times[station]  = []
        station_ranges[station] = []
        station_range_rates[station] = []

    times = np.arange(*time_arange_args)
    

    for et in times:

        for station in stations:
            moon_occult = spice.occult(str(object_id), 'POINT', ' ', '301', 'ELLIPSOID',
                                       'IAU_MOON', 'NONE', station, et)
            
            # compute observer-target state
            x0, lt0 = spice.spkcpo(str(object_id), et, station+'_TOPO', 'OBSERVER',
                                       'NONE', station_xyz[station], 'EARTH', 'ITRF93')
            x0 *= 1000.0

            unit = x0[0:3] / norm(x0[0:3])
            v_component = unit.dot(x0[3:6])
            

            # Get azimuth and elevation
            r, lon, lat = spice.reclat(x0[0:3])
            if moon_occult >= 0:
                if lat >= min_elevation:
                    station_times[station].append(et)
                    station_ranges[station].append(norm(x0[0:3]))
                    station_range_rates[station].append(v_component)
                else:
                    print("range = {} m, azimuth = {}, elevation = {}".format(r, lon * 180/np.pi, lat * 180/np.pi))
            else:
                print("moon occulting")

    return station_times, station_ranges, station_range_rates
