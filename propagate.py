from orekit_utils import *
from plot_ephemeris import *

from spice_loader import *

class WriteSpiceEphemerisHandler(PythonOrekitFixedStepHandler):
    center_id      = 399
    ref_frame_name = 'J2000'
    segment_id     = 'transit'
    degree         = 15
    write          = True
    
    def init(self, x0, t, step):
        import os
        if self.write and os.path.isfile(self.filename):
            os.remove(self.filename)
        self.x   = []
        self.t   = []

    def handleStep(self, x0, is_last):
        pv = x0.getPVCoordinates()
        t  = pv.getDate().durationFrom(AbsoluteDate.J2000_EPOCH)
        p  = pv.getPosition()
        v  = pv.getVelocity()
        self.x.append( np.array([p.getX(), p.getY(), p.getZ(),
                                 v.getX(), v.getY(), v.getZ()]) )
        self.t.append(t)

        #print(t)
        
        if is_last:
            self.x = np.vstack(self.x) / 1000.0
            self.t = np.array(self.t)

            # Open file and write it
            if self.write:
                spk = spice.spkopn(self.filename, "SPK_file", 0)
                spice.spkw13(spk, self.body_id, self.center_id, self.ref_frame_name, self.t[0], self.t[-1], self.segment_id, self.degree, self.t.shape[0], self.x, self.t)
                spice.spkcls(spk)
                
def create_propagator(t0, x0,
                      handler        = None,
                      min_step       = 0.001,
                      max_step       = 300.0,
                      rtol           = 1e-15,
                      atol           = 1e-9,
                      fixed_step     = 60.0,
                      dP             = 1e-3,
                      gravity_degree = 20,
                      gravity_order  = 20,
                      req            = None,
                      flattening     = None):
    gravity_field  = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    
    j2000          = FramesFactory.getEME2000()
    orbit          = CartesianOrbit(x0, j2000, t0, gravity_field.getMu())
    
    integrator     = DormandPrince853Integrator(min_step, max_step, rtol, atol) #prop_builder.buildPropagator()
    propagator     = NumericalPropagator(integrator)
    # itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    # body = OneAxisEllipsoid(req, flattening, itrf93)
    # propagator.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))
    
    propagator.setInitialState(SpacecraftState(orbit))
    propagator.setOrbitType(OrbitType.CARTESIAN)
    if handler is not None:
        propagator.setMasterMode(fixed_step, handler)

    return propagator
        
def propagate(t0, x0, tf,
              filename       = 'mission.bsp',
              write          = True):

    print("mu = {}".format(gravity_field.getMu()))

    eph_writer                = WriteSpiceEphemerisHandler()
    eph_writer.filename       = filename
    eph_writer.body_id        = object_id
    eph_writer.write          = write

    propagator = create_propagator(t0, x0, eph_writer, **kwargs)
    propagator.propagate(tf)

    return eph_writer
    
if __name__ == '__main__':

    spice_loader = SpiceLoader()
    
    # Initial state
    t0 = orekit_time(708687952.5569172)
    x0 = orekit_state([-6.45306258e+06, -1.19390257e+06, -8.56858164e+04,
                        1.83609046e+03, -9.56878337e+03, -4.95077925e+03])
    tf = orekit_time(709099110.5780709)


    eph = propagate(t0, x0, tf)

    spice_loader.clear()

    spice_loader = SpiceLoader('mission')
