from spice_loader import *
from orekit_utils import *
from propagate import propagate, WriteSpiceEphemerisHandler
import frames

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
min_step = 1e-15
max_step = 300.0
dP = 1.0

# For propagator
position_scale = dP

# Levenberg-Marquardt
bound_factor = 1e6


class LunarBatchLSObserver(PythonBatchLSObserver):
    def evaluationPerformed(self, iterations_count, evaluations_count, orbits,
                            estimated_orbital_parameters, estimated_propagator_parameters,
                            estimated_measurements_parameters, evaluations_provider,
                            lsp_evaluation):
        print("hi")
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
            ax.plot(xs[0,:], xs[1,:], xs[2,:], label="{}".format(iterations_count), alpha=min(1.0, 0.05 * iterations_count), c='r')
        except ZeroDivisionError:
            print("Warning: Couldn't plot due to zero division error")
        
        print("Test")




if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: python3 orekit_fit.py <prefix> <filename> <range_sigma> <range_rate_sigma>")
        exit
        
    prefix           = sys.argv[1]
    filename         = sys.argv[2]
    range_sigma      = float(sys.argv[3])
    range_rate_sigma = float(sys.argv[4])
    
    print("Processing '{}'...".format(filename))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], label='earth', alpha=0.5)

    dynamics = Dynamics()
    
    
    gravity_field = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)

    # We don't want to start at the beginning of the trajectory
    # because it's too hard to propagate through the low altitide
    # parts.
    et0, etf = loader.coverage()
    #et0 += 3600.0
    t0 = orekit_time(et0)
    x0 = orekit_state(spice.spkez(-5440, et0, 'J2000', 'NONE', 399)[0] * 1000.0)
    guess = CartesianOrbit(x0, j2000, t0, mu)

    # Setup ground stations
    station_names = ('DSS-23', 'DSS-33', 'DSS-53')
    station_data = orekit_spice_stations(body, et0, station_names)
    station_data, range_objs, rate_objs, azel_objs = orekit_test_data(body, filename, satellite, station_data,
                                                                      range_sigma      = range_sigma,
                                                                      range_rate_sigma = range_rate_sigma)
    print("Finished reading")
    if range_sigma == 0.0:
        print("Range/AzEl")
        measurements = range_objs + azel_objs
    elif range_rate_sigma == 0.0:
        print("RangeRate/AzEl")
        measurements = rate_objs + azel_objs
    else:
        print("Range/RangeRate/AzEl")
        measurements = range_objs + rate_objs + azel_objs

    
    #optimizer = GaussNewtonOptimizer(QRDecomposer(1e-11), False) #LevenbergMarquardtOptimizer()
    optimizer = LevenbergMarquardtOptimizer().withInitialStepBoundFactor(bound_factor)
    
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, dP)
    prop_builder = NumericalPropagatorBuilder(guess, integ_builder, PositionAngle.TRUE, position_scale)
    prop_builder.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    #prop_builder.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    
    estimator = BatchLSEstimator(optimizer, prop_builder)
    estimator.parametersConvergenceThreshold = 0.1
    estimator.maxIterations = 40
    estimator.maxEvaluations = 40

    for measurement in measurements:
        estimator.addMeasurement(measurement)

    estimator.setObserver(LunarBatchLSObserver())
    
    propagator = estimator.estimate()[0]
    logger = WriteSpiceEphemerisHandler()
    logger.body_id = -5440
    logger.write = False

    propagator.setMasterMode(300.0, logger)
    propagator.propagate(orekit_time(etf))
    xfit = logger.x.T * 1000.0
    
    ax.plot(xfit[0,:], xfit[1,:], xfit[2,:], alpha=0.5, label='fit')

    cov_inrtl = orekit_matrix_to_ndarray(estimator.getPhysicalCovariances(1e-12))

    # Get earth and moon-relative inertial states
    x_eci = logger.x[-1] * 1000.0
    xl_eci = spice.spkez(301, et0, 'J2000', 'NONE', 399)[0] * 1000.0
    x_lci = x_eci - xl_eci
    
    # Earth LVLH
    T_inrtl_to_elvlh = frames.compute_T_inrtl_to_lvlh(x_eci)
    cov_elvlh  = T_inrtl_to_elvlh.dot(cov_inrtl).dot(T_inrtl_to_elvlh.T)

    # Lunar LVLH
    T_inrtl_to_llvlh = frames.compute_T_inrtl_to_lvlh(x_lci)
    cov_llvlh = T_inrtl_to_llvlh.dot(cov_inrtl).dot(T_inrtl_to_llvlh.T)

    # Plot the moon's trajectory
    xls = []
    for et in np.arange(et0, etf, 3600.0):
        xls.append( spice.spkez(301, et, 'J2000', 'NONE', 399)[0] * 1000.0 )

    xls = np.vstack(xls).T
    ax.plot(xls[0,:], xls[1,:], xls[2,:], alpha=0.5, label='moon')
    
    ax.legend()

    print("Earth LVLH 3-sigma = {}".format(3.0 * np.sqrt(np.diag(cov_llvlh))))
    print("Lunar LVLH 3-sigma = {}".format(3.0 * np.sqrt(np.diag(cov_elvlh))))

    np.save("{}.{}.llvlh.npy".format(prefix, filename), cov_llvlh)
    np.save("{}.{}.elvlh.npy".format(prefix, filename), cov_elvlh)
    
    print("cov_llvlh = {}".format(cov_llvlh))
    print("cov_elvlh = {}".format(cov_elvlh))

    plt.show()
    #[0].getInitialState().getOrbit()

    
