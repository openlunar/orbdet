import orekit
vm = orekit.initVM()
print ('Java version:',vm.java_version)

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.utils import Constants, PVCoordinates, IERSConventions
#from org.orekit.estimation import Context
from org.orekit.estimation.leastsquares import BatchLSEstimator #, BatchLSObserver
from org.orekit.estimation.measurements import GroundStation, Range, RangeRate, PV, ObservedMeasurement, ObservableSatellite
from org.orekit.estimation.measurements.modifiers import Bias, OutlierFilter
from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer, GaussNewtonOptimizer
from org.hipparchus.linear import QRDecomposer
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
from org.orekit.models.earth.displacement import StationDisplacement
from org.orekit.time import TimeScalesFactory, AbsoluteDate
import org.orekit.time as oktime
from org.orekit.orbits import CartesianOrbit, PositionAngle
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, OceanTides, Relativity, SolidTides, ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory, NormalizedSphericalHarmonicsProvider
from org.orekit.propagation import Propagator
from org.orekit.propagation.conversion import NumericalPropagatorBuilder, DormandPrince853IntegratorBuilder
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.python import PythonBatchLSObserver

from java.util import ArrayList

from spice_loader import *
from generate import station_coords, generate_ground_measurements

loader = SpiceLoader('mission')

# Some global variables
mu = 398600435436095.9
j2000 = FramesFactory.getEME2000()
itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
station_names = ('DSS-15', 'DSS-45', 'DSS-65')
req = spice.bodvcd(399, 'RADII', 3)[1][0].item() * 1000.0
rpol = spice.bodvcd(399, 'RADII', 3)[1][2].item() * 1000.0
flattening = (req - rpol) / req
body = OneAxisEllipsoid(req, flattening, itrf93)
satellite = ObservableSatellite(0)
gravity_degree = 20
gravity_order  = 20


# For integrator
min_step = 0.001
max_step = 300.0
dP = 10.0

# For propagator
position_scale = dP

# For BLS
range_base_weight      = 1.0
range_rate_base_weight = 1.0


class StationData(object):
    def __init__(self, station_name, et,
                 range_sigma      = 20.0,
                 range_rate_sigma = 0.001):
        self.station          = orekit_station(station_name, et)
        self.range_sigma      = range_sigma
        self.range_rate_sigma = range_rate_sigma

class LunarBatchLSObserver(PythonBatchLSObserver):
    def evaluationPerformed(self, iterations_count, evaluations_count, orbits,
                            estimated_orbital_parameters, estimated_propagator_parameters,
                            estimated_measurements_parameters, evaluations_provider,
                            lsp_evaluation):
        drivers = estimated_orbital_parameters.getDrivers()
        print("{}:\t{} {} {}\t{} {} {}".format(iterations_count, drivers.get(0), drivers.get(1), drivers.get(2), drivers.get(3), drivers.get(4), drivers.get(5)))
        import pdb
        pdb.set_trace()
        print("Test")



def orekit_station(station_name, et):
    lon, lat, alt  = station_coords(station_name, et, req, flattening)
    pos            = GeodeticPoint(lat, lon, alt)
    frame_history  = FramesFactory.findEOP(body.getBodyFrame())
    topo_frame     = TopocentricFrame(body, pos, station_name)
    displacements  = []
    ground_station = GroundStation(topo_frame, frame_history, displacements)
    return ground_station

def orekit_time(ephemeris_time):
    return AbsoluteDate(AbsoluteDate.J2000_EPOCH, ephemeris_time)

def orekit_state(x0):
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    r = Vector3D(*(x0[0:3]))
    v = Vector3D(*(x0[3:6]))
    return PVCoordinates(r, v)

def orekit_measurements(measurements):
    ary = ArrayList().of_(ObservedMeasurement)
    for measurement in measurements:
        ary.add(measurement)
    return ary

def orekit_ranges(station_data, station_ets, station_ranges):
    orekit_ranges = []
    for station in station_ranges:
        ets = station_ets[station]
        for ii, rho in enumerate(station_ranges[station]):
            time = orekit_time(ets[ii].item())
            orekit_ranges.append( Range(station_data[station].station, True, time, rho, station_data[station].range_sigma, range_base_weight, satellite) )
    return orekit_ranges

def orekit_range_rates(station_data, station_ets, station_range_rates):
    orekit_range_rates = []
    for station in station_ranges:
        ets = station_ets[station]
        for ii,rho_dot in enumerate(station_range_rates[station]):
            time = orekit_time(ets[ii].item())
            orekit_range_rates.append( RangeRate(station_data[station].station, time, rho_dot, station_data[station].range_rate_sigma, range_rate_base_weight, True, satellite) )
    return orekit_range_rates



if __name__ == '__main__':


    et0, etf = SpiceLoader.spk_coverage('kernels/mission.bsp')
    print("et0, etf = {}, {}".format(et0, etf))

    # Cut off the ends to avoid light time problems
    et0 += 100.0
    etf -= 100.0
    
    t0  = orekit_time(et0)
    x0  = orekit_state([-6.45306258e+06, -1.19390257e+06, -8.56858164e+04,
                         1.83609046e+03, -9.56878337e+03, -4.95077925e+03])

    # Generate measurements
    station_ets, station_ranges, station_range_rates, station_elevations = generate_ground_measurements('mission', -5440, station_names, (et0, (etf + et0) * 0.5, 1000.0))
    
    # Setup ground stations
    station_data = {}
    for name in station_names:
        station_data[name] = StationData(name, et0)
    
    # Put measurements into orekit Range and RangeRate objects (in a Python list)
    range_objs = orekit_ranges(station_data, station_ets, station_ranges)
    range_rate_objs = orekit_range_rates(station_data, station_ets, station_range_rates)
    measurements = range_objs + range_rate_objs
    #measurements = orekit_measurements(range_objs + range_rate_objs)
    
    gravity_field = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    guess = CartesianOrbit(x0, j2000, t0, gravity_field.getMu())
   
    
    optimizer = GaussNewtonOptimizer(QRDecomposer(1e-11), False) #LevenbergMarquardtOptimizer()

    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, dP)
    prop_builder = NumericalPropagatorBuilder(guess, integ_builder, PositionAngle.TRUE, position_scale)
    #prop_builder.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    
    estimator = BatchLSEstimator(optimizer, prop_builder)
    estimator.parametersConvergenceThreshold = 1e-3
    estimator.maxIterations = 10
    estimator.maxEvaluations = 20

    for measurement in measurements:
        estimator.addMeasurement(measurement)


    estimator.setObserver(LunarBatchLSObserver())
    estimated_orbit = estimator.estimate()[0].getInitialState().getOrbit()

    
    
    #builder   = NumericalPropagatorBuilder(guess, frame, )

    print (Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
