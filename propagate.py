from orekit_utils import *
from plot_ephemeris import *

from spice_loader import *


class NoMapperError(ValueError):
    pass


class WriteSpiceEphemerisHandler(PythonOrekitFixedStepHandler):
    center_id      = 399
    ref_frame_name = 'J2000'
    segment_id     = 'transit'
    degree         = 15
    write          = True
    mapper         = None
    
    def init(self, x0, t, step):
        import os
        if self.write and os.path.isfile(self.filename):
            os.remove(self.filename)
        self.x    = []
        self.t    = []
        self.jPhi = Array2DRowRealMatrix(6,6)

    def handleStep(self, x0, is_last):
        pv = x0.getPVCoordinates()
        t  = pv.getDate().durationFrom(AbsoluteDate.J2000_EPOCH)
        p  = pv.getPosition()
        v  = pv.getVelocity()
        
        self.x.append( np.array([p.getX(), p.getY(), p.getZ(),
                                 v.getX(), v.getY(), v.getZ()]) )
        self.t.append(t)
        
        if is_last:
            print("is_last")
            self.x = np.vstack(self.x) / 1000.0
            self.t = np.array(self.t)

            # Open file and write it
            if self.write:
                spk = spice.spkopn(self.filename, "SPK_file", 0)
                spice.spkw13(spk, self.body_id, self.center_id, self.ref_frame_name, self.t[0], self.t[-1], self.segment_id, self.degree, self.t.shape[0], self.x, self.t)
                spice.spkcls(spk)

        if self.mapper is None:
            raise NoMapperError("no mapper defined in handler")
        self.mapper.getStateJacobian(x0, self.jPhi.getDataRef())
        #print(self.Phi)
        
    @property
    def Phi(self):
        return orekit_matrix_to_ndarray(self.jPhi)
                
def create_propagator(t0, x0,
                      handler        = None,
                      min_step       = 0.001,
                      max_step       = 300.0,
                      rtol           = 1e-15,
                      atol           = 1e-9,
                      fixed_step     = 60.0,
                      dP             = None,
                      gravity_degree = 20,
                      gravity_order  = 20,
                      req            = None,
                      flattening     = None):
    gravity_field  = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    
    j2000          = FramesFactory.getEME2000()
    orbit          = CartesianOrbit(x0, j2000, t0, gravity_field.getMu())

    if dP is not None: # Compute absolute and relative tolerances
        tols = NumericalPropagator.tolerances(dP, orbit, OrbitType.CARTESIAN)
        atol = orekit.JArray_double.cast_(tols[0])
        rtol = orekit.JArray_double.cast_(tols[1])

    integrator     = DormandPrince853Integrator(min_step, max_step, atol, rtol) #prop_builder.buildPropagator()
    propagator     = NumericalPropagator(integrator)
    propagator.addForceModel(NewtonianAttraction(gravity_field.getMu()))
    propagator.setOrbitType(OrbitType.CARTESIAN)
    #propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    # itrf93 = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    # body = OneAxisEllipsoid(req, flattening, itrf93)
    # propagator.addForceModel(HolmesFeatherstoneAttractionModel(body.getBodyFrame(), gravity_field))

    pde = PartialDerivativesEquations("dYdY0", propagator)
    initial_state = pde.setInitialJacobians(SpacecraftState(orbit))
    
    propagator.setInitialState(initial_state)

    if handler is not None:
        handler.mapper = pde.getMapper()
        propagator.setMasterMode(fixed_step, handler)

    return propagator, pde
        
def propagate(t0, x0, tf,
              object_id       = -5440,
              filename        = 'mission.bsp',
              write_ephemeris = False,
              **kwargs):

    eph_writer                = WriteSpiceEphemerisHandler()
    eph_writer.filename       = filename
    eph_writer.body_id        = object_id
    eph_writer.write          = write_ephemeris
    
    propagator, pde = create_propagator(t0, x0, eph_writer, **kwargs)

    final_state = propagator.propagate(tf)

    #jPhi = Array2DRowRealMatrix(6,6)
    #pde.getMapper().getStateJacobian(final_state, jPhi.getDataRef())
    #Phi = orekit_matrix_to_ndarray(eph_writer.Phi)

    return eph_writer, final_state
    
if __name__ == '__main__':

    spice_loader = SpiceLoader()
    moon = CelestialBodyFactory.getBody("MOON")
    j2000 = FramesFactory.getEME2000()
    
    # Initial state
    #t0 = AbsoluteDate(DateTimeComponents.parseDateTime("2019-04-04T01:37:00Z"), TimeScalesFactory.getUTC())
    #tf = AbsoluteDate(DateTimeComponents.parseDateTime("2019-04-04T17:37:00Z"), TimeScalesFactory.getUTC())
    t0 = orekit_time(708687952.5569172)
    tf = orekit_time(709099110.5780709)
    #tf = orekit_time(708689252.5569172)
    
    #x0_ = np.array([384730.575243, 58282.200599, -5689.089133,
    #                0.238079, 0.158155, 0.055987]) * 1000.0
    x0_ = np.array([-6.45306258e+06, -1.19390257e+06, -8.56858164e+04,
                     1.83609046e+03, -9.56878337e+03, -4.95077925e+03])
    dx1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x01_ = x0_ + dx1
    
    #deltav = np.array([ 526.82780975, -2745.5625324, -1420.52270256])
    #deltav_mag = norm(deltav)
    #u_deltav = deltav / deltav_mag
    #x0_pre = np.array(x0_)
    #x0_pre[3:6] -= delta_v
    #x0_post = x0_pre + T_misalign.dot(delta_v)
    
    x0 = orekit_state(x0_)
    x01 = orekit_state(x01_)
    

    eph, xf = propagate(t0, x0, tf, write_ephemeris = True, dP = 0.001)
    eph1, xf1 = propagate(t0, x01, tf, write_ephemeris = False, dP = 0.001)
    Phi = eph.Phi

    dxf_pred = Phi.dot(dx1)
    dxf      = (eph1.x[-1,:] - eph.x[-1,:]) * 1000.0

    print("dxf      = {}".format(dxf))
    print("dxf pred = {}".format(dxf_pred))
    print("Phi      = {}".format(Phi))
    

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    xs = eph.x * 1000.0
    axes.plot(xs[:,0], xs[:,1], xs[:,2], alpha=0.5, label='sc')
    axes.scatter([0.0], [0.0], [0.0], label='earth')

    xl = []
    t = t0
    while t.durationFrom(tf) < 0:
        tx = PVCoordinatesProvider.cast_(moon).getPVCoordinates(t, j2000)
        r = tx.getPosition()
        xl.append([r.getX(), r.getY(), r.getZ()])

        t = AbsoluteDate(t, 600.0)

    xl = np.vstack(xl)
    axes.plot(xl[:,0], xl[:,1], xl[:,2], alpha=0.5, label='moon')
    #print("Lunar initial state: {}".format(xl[0,:]))
    #print("Lunar final state: {}".format(xl[-1,:]))
    #print("Num lunar states: {}".format(xl.shape[0]))
    #print("S/c final state: {}".format(xs[-1,:]))

    plt.show()

    

    spice_loader.clear()

    spice_loader = SpiceLoader('mission')
