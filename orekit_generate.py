from orekit_utils import *
from propagate import *

from org.orekit.estimation.measurements.generation import Generator, RangeBuilder, RangeRateBuilder

from spice_loader import *

import sys

def generate_measurements(station, station_name, meas_type, et0, etf, x0,
                        sigma       = None,
                        base_weight = 1.0,
                        two_way     = True,
                        step        = 2400.0,
                        seed        = 0):
    t0 = orekit_time(et0)
    #print(t0.toString())
    tf = orekit_time(etf)
    state0 = orekit_state(x0)
    
    noise_source = orekit_gaussian_vector_generator(seed, sigma = sigma)
    propagator, pde = create_propagator(t0, state0)

    generator = Generator()
    satellite = generator.addPropagator(propagator)

    if meas_type == 'RANGE':
        builder_class = RangeBuilder
    elif meas_type == 'RATE':
        builder_class = RangeRateBuilder
    builder = builder_class(noise_source, station, two_way, sigma, base_weight, satellite)
    
    fixed_step_selector = FixedStepSelector(step, TimeScalesFactory.getUTC())
    elevation_detector  = ElevationDetector(station.getBaseFrame()).withConstantElevation(5.0 * math.pi/180.0).withHandler(ContinueOnEvent())
    #eclipse_detector    = EclipseDetector(satellite, 0.5, CelestialBodyFactory.getMoon())
    scheduler = EventBasedScheduler(builder, fixed_step_selector, propagator, elevation_detector, SignSemantic.FEASIBLE_MEASUREMENT_WHEN_POSITIVE)
    generator.addScheduler(scheduler)

    measurements = generator.generate(t0, tf)
    for meas_obj in measurements:
        if meas_type == 'RANGE':
            meas = Range.cast_(meas_obj)
        elif meas_type == 'RATE':
            meas = RangeRate.cast_(meas_obj)
        else:
            raise ValueError("unrecognized meas_type")

        if two_way:
            way_str = "TWOWAY"
        else:
            way_str = "ONEWAY"
        print("{}   {} {}       {}        {}".format(meas.getDate().toString(), way_str, meas_type, station_name, meas.getObservedValue()[0]))
    
    
    

if __name__ == '__main__':
    
    range_sigma = 2.0
    range_rate_sigma = 0.001
    base_weight = 1.0
    seed = int(sys.argv[1])
    step = float(sys.argv[2])
    if sys.argv[3] == '1':
        range_two_way = False
    else:
        range_two_way = True
    if sys.argv[4] == '1':
        rate_two_way = False
    else:
        rate_two_way = True

    loader = SpiceLoader('mission')

    itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    req = spice.bodvcd(399, 'RADII', 3)[1][0].item() * 1000.0
    rpol = spice.bodvcd(399, 'RADII', 3)[1][2].item() * 1000.0
    flattening = (req - rpol) / req
    body = OneAxisEllipsoid(req, flattening, itrf93)
    

    et0, etf = loader.coverage()
    #et0 += 3600.0 # skip the really difficult to propagate through part of the trajectory
    x0 = spice.spkez(-5440, et0, 'J2000', 'NONE', 399)[0] * 1000.0
    #print(et0)
    #print(x0)
    
    station_names = ('DSS-23', 'DSS-33', 'DSS-53') # 11m DSN dishes
    stations = orekit_spice_stations(body, et0, station_names)
    #stations = orekit_test_stations(body) # dishes used in Orekit's example
    
    for station_name in stations:
        station = stations[station_name]
        
        generate_measurements(station, station_name, 'RANGE', et0, etf, x0,
                              sigma = range_sigma, two_way = range_two_way, seed = seed, step = step)
        generate_measurements(station, station_name, 'RATE', et0, etf, x0,
                              sigma = range_rate_sigma, two_way = rate_two_way, seed = seed + 1, step = step)

