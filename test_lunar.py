from spice_loader import *
from generate import station_coords, generate_ground_measurements
from orekit_utils import *
from plot_ephemeris import plot_ephemeris
from propagate import propagate

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
#mu = 398600435436095.9
j2000 = FramesFactory.getEME2000()
itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
station_names = ('DSS-15', 'DSS-45', 'DSS-65')
req = spice.bodvcd(399, 'RADII', 3)[1][0].item() * 1000.0
rpol = spice.bodvcd(399, 'RADII', 3)[1][2].item() * 1000.0
print("req = {}".format(req))
print("rpol = {}".format(rpol))

flattening = (req - rpol) / req
print("f = {}".format(flattening))
body = OneAxisEllipsoid(req, flattening, itrf93)
satellite = ObservableSatellite(0)
gravity_degree = 20
gravity_order  = 20


# For integrator
min_step = 0.001
max_step = 300.0
dP = 0.1

# For propagator
position_scale = dP

# Levenberg-Marquardt
bound_factor = 1e8



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
        earth_moon_state[6:12] = spice.spkez(301, et0, 'J2000', 'NONE', 399)[0] * 1000.0
        earth_moon_state[12:] = np.identity(6).reshape(36)

        print("Trying to plot...")

        t0 = orbits[0].date
        x0 = orekit_state(state)
        tf = orekit_time(self.tf)
        
        eph = propagate(t0, x0, tf, write = False)

        ax.plot(eph.x[:,0] * 1000.0, eph.x[:,1] * 1000.0, eph.x[:,2] * 1000.0, label="{}".format(iterations_count), alpha=(1/40.0) * iterations_count, c='r')
        #except ZeroDivisionError:
        #    print("Warning: Couldn't plot due to zero division error")
        

        
if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], label='earth')
    
    dynamics = Dynamics()
    et0, etf = SpiceLoader.spk_coverage('kernels/mission.bsp')
    print("et0, etf = {}, {}".format(et0, etf))

    # Cut off the ends to avoid light time problems
    et0 += 100.0
    etf -= 100.0
    
    t0  = orekit_time(et0)
    x0  = orekit_state([-6.45306258e+06, -1.19390257e+06, -8.56858164e+04,
                         1.83609046e+03, -9.56878337e+03, -4.95077925e+03])
    #x0 = PVCoordinates(Vector3D(-40517522.9, -10003079.9, 166792.8),
    #                   Vector3D(762.559, -1474.468, 55.430))

    # Generate measurements
    station_ets, station_ranges, station_range_rates, station_elevations = generate_ground_measurements('mission', -5440, station_names, (et0, etf, 10000.0))
    
    # Setup ground stations
    station_data = orekit_spice_stations(body, station_names, et0)
    
    # Put measurements into orekit Range and RangeRate objects (in a Python list)
    range_objs = orekit_ranges(satellite, station_data, station_ets, station_ranges)
    range_rate_objs = orekit_range_rates(satellite, station_data, station_ets, station_range_rates)
    measurements = range_rate_objs
    #measurements = orekit_measurements(range_objs + range_rate_objs)
    
    gravity_field = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    guess = CartesianOrbit(x0, j2000, t0, gravity_field.getMu())
   
    
    #optimizer = GaussNewtonOptimizer(QRDecomposer(1e-11), False) #LevenbergMarquardtOptimizer()
    optimizer = LevenbergMarquardtOptimizer().withInitialStepBoundFactor(bound_factor)
    
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, dP)
    prop_builder = NumericalPropagatorBuilder(guess, integ_builder, PositionAngle.TRUE, position_scale)
    #prop_builder.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    
    estimator = BatchLSEstimator(optimizer, prop_builder)
    estimator.parametersConvergenceThreshold = 1e-3
    estimator.maxIterations = 40
    estimator.maxEvaluations = 40

    for measurement in measurements:
        estimator.addMeasurement(measurement)


    observer = LunarBatchLSObserver()
    observer.tf = etf
    
    estimator.setObserver(observer)
    try:
        estimated_orbit = estimator.estimate() #[0].getInitialState().getOrbit()
    except:
        for ii,et in enumerate(np.arange(et0, etf, (etf - et0) / 20.0)):
            rm = spice.spkezp(301, et, 'J2000', 'NONE', 399)[0]
            ax.scatter([rm[0]], [rm[1]], [rm[2]], c='b', alpha = ii/20.0, s=2)

        spice_loader = SpiceLoader('mission')
        plot_ephemeris(spice_loader, axes = ax)
        
        plt.show()
            
