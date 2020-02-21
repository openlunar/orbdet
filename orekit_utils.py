import orekit
vm = orekit.initVM()
print ('Java version:',vm.java_version)

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.utils import Constants, PVCoordinates, IERSConventions
#from org.orekit.estimation import Context
from org.orekit.estimation.leastsquares import BatchLSEstimator #, BatchLSObserver
from org.orekit.estimation.measurements import GroundStation, Range, RangeRate, AngularAzEl, PV, ObservedMeasurement, ObservableSatellite
from org.orekit.estimation.measurements.modifiers import Bias, OutlierFilter
from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer, GaussNewtonOptimizer
from org.hipparchus.linear import QRDecomposer
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
from org.orekit.models.earth.displacement import StationDisplacement
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateTimeComponents
import org.orekit.time as oktime
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit, EquinoctialOrbit, PositionAngle
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, OceanTides, Relativity, SolidTides, ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory, NormalizedSphericalHarmonicsProvider
from org.orekit.propagation import Propagator
from org.orekit.propagation.conversion import NumericalPropagatorBuilder, DormandPrince853IntegratorBuilder
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.python import PythonBatchLSObserver

from org.hipparchus.geometry.euclidean.threed import Vector3D

from java.util import ArrayList

import math
from generate import station_coords

OREKIT_TEST_STATIONS = {
    'Uralla':     ( -30.632947613,   151.5650529068, 1163.2667864364 ),
    'Kumsan':     (  36.1247623774,  127.4871671976,  180.5488660489 ),
    'Pretoria':   ( -25.8854896226,   27.7074493158, 1566.6334663324 ),
    'CastleRock': (  39.2764477379, -104.8063531025, 2095.3769797949 ),
    'Fucino':     (  41.9774962512,   13.6004229863,  671.3542005921 )
    }

class StationData(object):
    def __init__(self, body, station_name,
                 et               = None,
                 range_sigma      = 20.0,
                 range_rate_sigma = 0.001):
        if station_name in OREKIT_TEST_STATIONS:
            self.station      = orekit_station_by_coords(body,
                                                         *(OREKIT_TEST_STATIONS[station_name]))
        else:
            self.station          = orekit_station(body, station_name, et)
        self.range_sigma      = range_sigma
        self.range_rate_sigma = range_rate_sigma

def orekit_station_by_geodetic_point(body, station_name, pos, displacements = []):
    frame_history  = FramesFactory.findEOP(body.getBodyFrame())
    topo_frame     = TopocentricFrame(body, pos, station_name)
    ground_station = GroundStation(topo_frame, frame_history, displacements)
    return ground_station

def orekit_station_by_coords(body, station_name, lat, lon, alt, displacements = []):
    pos = GeodeticPoint(lat, lon, alt)
    frame_history  = FramesFactory.findEOP(body.getBodyFrame())
    topo_frame     = TopocentricFrame(body, pos, station_name)
    displacements  = []
    ground_station = GroundStation(topo_frame, frame_history, displacements)
    return ground_station

def orekit_station(body, station_name, et,
                   displacements = [],
                   req           = 6378136.6,
                   flattening    = 0.0033528131084554157):
    lon, lat, alt  = station_coords(station_name, et, req, flattening)
    pos            = GeodeticPoint(lat, lon, alt)
    return orekit_station_by_geodetic_point(body, station_name, pos, displacements)

def orekit_spice_stations(body, station_names, et):
    station_data = {}
    for name in station_names:
        station_data[name] = StationData(body, name, et)
    return station_data

def orekit_test_stations(body):
    stations = {}
    for station in OREKIT_TEST_STATIONS:
        lat, lon, alt = OREKIT_TEST_STATIONS[station]
        stations[station] = orekit_station_by_coords(body, station, lat, lon, alt)
    return stations

def orekit_drivers_to_values(ds):
    import numpy as np
    ary = []
    for ii in range(0, ds.size()):
        ary.append(ds.get(ii).getValue())
    return np.array(ary)

def orekit_time(ephemeris_time):
    return AbsoluteDate(AbsoluteDate.J2000_EPOCH, ephemeris_time)

def orekit_state(x0):
    r = Vector3D(*(x0[0:3]))
    v = Vector3D(*(x0[3:6]))
    return PVCoordinates(r, v)

def orekit_measurements(measurements):
    ary = ArrayList().of_(ObservedMeasurement)
    for measurement in measurements:
        ary.add(measurement)
    return ary

def orekit_ranges(satellite, station_data, station_ets, station_ranges,
                  range_base_weight = 1.0):
    orekit_ranges = []
    for station in station_ranges:
        ets = station_ets[station]
        for ii, rho in enumerate(station_ranges[station]):
            time = orekit_time(ets[ii].item())
            orekit_ranges.append( Range(station_data[station].station, True, time, rho, station_data[station].range_sigma, range_base_weight, satellite) )
    return orekit_ranges

def orekit_range_rates(satellite, station_data, station_ets, station_range_rates,
                       range_rate_base_weight = 1.0):
    orekit_range_rates = []
    for station in station_range_rates:
        ets = station_ets[station]
        for ii,rho_dot in enumerate(station_range_rates[station]):
            time = orekit_time(ets[ii].item())
            orekit_range_rates.append( RangeRate(station_data[station].station, time, rho_dot, station_data[station].range_rate_sigma, range_rate_base_weight, True, satellite) )
    return orekit_range_rates

def orekit_test_data(body, filename, satellite):
    """Load test data from W3B.aer"""
    azels = []
    ranges = []

    stations = orekit_test_stations(body)
    
    f = open(filename, 'r')
    for line in f:
        if line[0] == '#':
            continue
        elif len(line) < 10:
            continue

        fields = line.split()
        date_components = DateTimeComponents.parseDateTime(fields[0])
        date = AbsoluteDate(date_components, TimeScalesFactory.getUTC())
        mode = fields[1]
        station_name = fields[2]
        station = stations[station_name]
        
        if mode == 'RANGE':
            rho = float(fields[3])
            range_obj = Range(station, True, date, rho, 20.0, 1.0, satellite)
            ranges.append( range_obj )
        elif mode == 'AZ_EL':
            az = float(fields[3]) * math.pi / 180.0
            el = float(fields[4]) * math.pi / 180.0
            azel_obj = AngularAzEl(station, date, [az, el], [0.02, 0.02], [1.0, 1.0], satellite)
            azels.append( azel_obj )
        else:
            raise SyntaxError("unrecognized mode '{}'".format(mode))

    return stations, ranges, azels
