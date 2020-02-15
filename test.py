import orekit
vm = orekit.initVM()
print ('Java version:',vm.java_version)

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.utils import Constants, PVCoordinates, IERSConventions
#from org.orekit.estimation import Context
from org.orekit.estimation.leastsquares import BatchLSEstimator #, BatchLSObserver
from org.orekit.estimation.measurements import GroundStation, Range, RangeRate, PV, ObservedMeasurement
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
from org.orekit.python import PythonBatchLSObserver as BatchLSObserver

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

class LunarBatchLSObserver(BatchLSObserver):
    def evaluationPerformed(iterations_count, evaluations_count, orbits,
                            estimated_orbital_parameters, estimated_propagator_parameters,
                            estimated_measurements_parameters, evaluations_provider,
                            lsp_evaluation):
        print("Test")



def orekit_station(station_name, et):
    lon, lat, alt  = station_coords(station_name, et)
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


if __name__ == '__main__':


    et0, etf = SpiceLoader.spk_coverage('kernels/mission.bsp')
    print("et0, etf = {}, {}".format(et0, etf))

    # Cut off the ends to avoid light time problems
    et0 += 100.0
    etf -= 100.0
    
    t0  = orekit_time(et0)
    x0  = orekit_state([-6.45306258e+06, -1.19390257e+06, -8.56858164e+04, 1.83609046e+03, -9.56878337e+03, -4.95077925e+03])

    ets, ranges, range_rates = generate_ground_measurements('mission', -5440, station_names, (et0, et0 + 10000.0, 1.0))
    import pdb
    pdb.set_trace()
    
    gravity_field = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    guess = CartesianOrbit(x0, j2000, t0, gravity_field.getMu())
   
    
    optimizer = GaussNewtonOptimizer(QRDecomposer(1e-11), False) #LevenbergMarquardtOptimizer()

    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, dP)
    prop_builder = NumericalPropagatorBuilder(guess, integ_builder, PositionAngle.MEAN, position_scale)
    prop_builder.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    
    estimator = BatchLSEstimator(optimizer, prop_builder)
    estimator.parametersConvergenceThreshold = 1e-3
    estimator.maxIterations = 10
    estimator.maxEvaluations = 20

    # Setup ground stations
    station_data = {}
    for name in station_names:
        station_data[name] = StationData(name, et0)

    range_test = Range(station_data['DSS-15'].station, True, t0, 1500000.0, station_data['DSS-15'].range_sigma, range_base_weight, None)
    range_rate_test = RangeRate(station_data['DSS-15'].station, t0, 5000.0, station_data['DSS-15'].range_rate_sigma, range_rate_base_weight, True, None)

    measurements = orekit_measurements([range_test, range_rate_test])

    estimator.setObserver(LunarBatchLSObserver())
    estimated_orbit = estimator.estimate()[0].getInitialState().getOrbit()

    
    
    #builder   = NumericalPropagatorBuilder(guess, frame, )

    print (Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
