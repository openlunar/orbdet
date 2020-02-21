from spice_loader import *
from generate import station_coords, generate_ground_measurements
from orekit_utils import *

from scipy.linalg import norm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trajectory
from trajectory.propagate import Dynamics, propagate_to
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

loader = SpiceLoader('mission')

# Some global variables
mu = 398600435436095.9
j2000 = FramesFactory.getEME2000()
itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
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

# Levenberg-Marquardt
bound_factor = 1e6


class LunarBatchLSObserver(PythonBatchLSObserver):
    def evaluationPerformed(self, iterations_count, evaluations_count, orbits,
                            estimated_orbital_parameters, estimated_propagator_parameters,
                            estimated_measurements_parameters, evaluations_provider,
                            lsp_evaluation):
        drivers = estimated_orbital_parameters.getDrivers()

        state = orekit_drivers_to_values(drivers)
        print("{}:\t{} {} {}\t{} {} {}".format(iterations_count, *state))
        
        print("r = {}\tv = {}".format(norm(state[0:3]), norm(state[3:6])))

        earth_moon_state = np.zeros(48)
        earth_moon_state[0:6] = state
        earth_moon_state[6:12] = np.array([384402000.0, 0.0, 0.0,
                                           0.0, 2.649e-6 * 384402000.0, 0.0])
        earth_moon_state[12:] = np.identity(6).reshape(36)

        print("Trying to plot...")
        try:
            ts, xs, xf, Phi = propagate_to(dynamics, 0.0, earth_moon_state, 30000.0,
                                           max_step = 500.0)
            ax.plot(xs[0,:], xs[1,:], xs[2,:], label="{}".format(iterations_count), alpha=0.05 * iterations_count, c='r')
        except ZeroDivisionError:
            print("Warning: Couldn't plot due to zero division error")

        if iterations_count == 20:
            plt.show()
        
        print("Test")




if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], label='earth')

    dynamics = Dynamics()
    

    # Setup ground stations
    station_data = orekit_test_stations(body)
    station_data, range_objs, azel_objs = orekit_test_data(body, 'W3B.aer', satellite)
    
    measurements = range_objs
    
    gravity_field = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)

    #guess = KeplerianOrbit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    #guess = EquinoctialOrbit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    guess_date_components = DateTimeComponents.parseDateTime("2010-11-02T02:56:15.690")
    guess_date = AbsoluteDate(guess_date_components, TimeScalesFactory.getUTC())
    guess = CartesianOrbit(PVCoordinates(Vector3D(-40517522.9, -10003079.9, 166792.8),
                                         Vector3D(762.559, -1474.468, 55.430)),
                           j2000, guess_date, mu)
    
    #optimizer = GaussNewtonOptimizer(QRDecomposer(1e-11), False) #LevenbergMarquardtOptimizer()
    optimizer = LevenbergMarquardtOptimizer().withInitialStepBoundFactor(bound_factor)
    
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, dP)
    prop_builder = NumericalPropagatorBuilder(guess, integ_builder, PositionAngle.TRUE, position_scale)
    #prop_builder.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    
    estimator = BatchLSEstimator(optimizer, prop_builder)
    estimator.parametersConvergenceThreshold = 1e-3
    estimator.maxIterations = 20
    estimator.maxEvaluations = 25

    for measurement in measurements:
        estimator.addMeasurement(measurement)


    estimator.setObserver(LunarBatchLSObserver())
    estimated_orbit = estimator.estimate()
    plt.show()
    #[0].getInitialState().getOrbit()

    
