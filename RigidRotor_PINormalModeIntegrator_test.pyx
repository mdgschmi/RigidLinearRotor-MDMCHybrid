# This module implements path integral MD integrator using normal mode coordinates
#
# Written by Konrad Hinsen
#

#cython: boundscheck=False, wraparound=False, cdivision=True

"""
Path integral MD integrator using normal-mode coordinates
"""

__docformat__ = 'restructuredtext'

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New

from libc.stdint cimport int32_t
import numpy as N
cimport numpy as N

from MMTK import Units, ParticleProperties, Features, Environment, Vector
import MMTK.PIIntegratorSupport
cimport MMTK.PIIntegratorSupport
import numbers

from MMTK.forcefield cimport energy_data
cimport MMTK.mtrand

include 'MMTK/trajectory.pxi'

cdef extern from "fftw3.h":
    ctypedef struct fftw_complex
    ctypedef void *fftw_plan
    cdef int FFTW_FORWARD, FFTW_BACKWARD, FFTW_ESTIMATE
    cdef void fftw_execute(fftw_plan p)
    cdef fftw_plan fftw_plan_dft_1d(int n, fftw_complex *data_in, fftw_complex *data_out,
                                    int sign, int flags)
    cdef void fftw_destroy_plan(fftw_plan p)

cdef extern from "stdlib.h":
    cdef double fabs(double)
    cdef double sqrt(double)
    cdef double sin(double)
    cdef double cos(double)
    cdef double exp(double)
    cdef double M_PI

cdef extern from "time.h":
    ctypedef unsigned long clock_t
    cdef clock_t clock()
    cdef enum:
        CLOCKS_PER_SEC

cdef double hbar = Units.hbar
cdef double k_B = Units.k_B

cdef bytes PLAN_CAPSULE_NAME = b'plan_capsule'

cdef void plan_capsule_destructor(object cap):
    fftw_destroy_plan(PyCapsule_GetPointer(cap, PLAN_CAPSULE_NAME))

#
# Velocity Verlet integrator in normal-mode coordinates
#
cdef class RigidRotor_PINormalModeIntegrator(MMTK.PIIntegratorSupport.PIIntegrator):

    """
    Molecular dynamics integrator for path integral systems using
    normal-mode coordinates.

    The integration is started by calling the integrator object.
    All the keyword options (see documentation of __init__) can be
    specified either when creating the integrator or when calling it.

    The following data categories and variables are available for
    output:

     - category "time": time

     - category "configuration": configuration and box size (for
       periodic universes)

     - category "velocities": atomic velocities

     - category "gradients": energy gradients for each atom

     - category "energy": potential and kinetic energy, plus
       extended-system energy terms if a thermostat and/or barostat
       are used

     - category "thermodynamic": temperature

     - category "auxiliary": primitive and virial quantum energy estimators

    """

    cdef N.ndarray workspace1, workspace2
    cdef double *workspace_ptr_1
    cdef double *workspace_ptr_2
    cdef dict plans
    cdef N.ndarray densmat, rotengmat
    cdef double rotmove
    cdef int rotstepskip

    def __init__(self, universe, **options):
        """
        :param universe: the universe on which the integrator acts
        :type universe: MMTK.Universe
        :keyword steps: the number of integration steps (default is 100)
        :type steps: int
        :keyword delta_t: the time step (default is 1 fs)
        :type delta_t: float
        :keyword actions: a list of actions to be executed periodically
                          (default is none)
        :type actions: list
        :keyword threads: the number of threads to use in energy evaluation
                          (default set by MMTK_ENERGY_THREADS)
        :type threads: int
        :keyword background: if True, the integration is executed as a
                             separate thread (default: False)
        :type background: bool
        """
        MMTK.PIIntegratorSupport.PIIntegrator.__init__(
            self, universe, options, "Path integral normal-mode integrator")
        # Supported features: PathIntegrals
        self.features = [Features.PathIntegralsFeature]

    default_options = {'first_step': 0, 'steps': 100, 'delta_t': 1.*Units.fs,
                       'background': False, 'threads': None,
                       'frozen_subspace': None, 'actions': []}

    available_data = ['time', 'configuration', 'velocities', 'gradients',
                      'energy', 'thermodynamic', 'auxiliary']

    restart_data = ['configuration', 'velocities', 'energy']

    # The implementation of the equations of motion follows the article
    #   Ceriotti et al., J. Chem. Phys. 133, 124104 (2010)
    # with the following differences:
    # 1) All the normal mode coordinates are larger by a factor sqrt(nbeads),
    #    and the non-real ones (k != 0, k != n/2) are additionally smaller by
    #    sqrt(2).
    # 2) The spring energy is smaller by a factor of nbeads to take
    #    into account the factor nbeads in Eq. (3) of the paper cited above.
    #    The potential energy of the system is also smaller by a factor of
    #    nbeads compared to the notation in this paper.
    # 3) Velocities are used instead of momenta in the integrator.
    # 4) Eq. (18) is also used for odd n, ignoring the k = n/2 case.

    cdef cartesianToNormalMode(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] nmc,
                               Py_ssize_t bead_index, int32_t nb):
        cdef double *w1 = self.workspace_ptr_1
        cdef double *w2 = self.workspace_ptr_2
        cdef fftw_plan p
        cdef Py_ssize_t i, j
        if nb == 1:
            for i in range(3):
                nmc[i, bead_index] = x[bead_index, i]
        else:
            try:
                p = PyCapsule_GetPointer(self.plans[(FFTW_FORWARD, nb)], PLAN_CAPSULE_NAME)
            except KeyError:
                p = fftw_plan_dft_1d(nb, <fftw_complex *>w1, <fftw_complex *>w2,
                                     FFTW_FORWARD, FFTW_ESTIMATE)
                self.plans[(FFTW_FORWARD, nb)] = \
                        PyCapsule_New(p, PLAN_CAPSULE_NAME, plan_capsule_destructor)
            for i in range(3):
                for j in range(nb):
                    w1[2*j] = x[bead_index+j, i]
                    w1[2*j+1] = 0.
                fftw_execute(p)
                nmc[i, bead_index+0] = w2[0]
                for j in range(1, (nb+1)/2):
                    nmc[i, bead_index+j] = w2[2*j]
                    nmc[i, bead_index+nb-j] = w2[2*j+1]
                if nb % 2 == 0:
                    nmc[i, bead_index+nb/2] = w2[nb]

    cdef normalModeToCartesian(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] nmc,
                               Py_ssize_t bead_index, int32_t nb):
        cdef double *w1 = self.workspace_ptr_1
        cdef double *w2 = self.workspace_ptr_2
        cdef fftw_plan p
        cdef Py_ssize_t i, j
        if nb == 1:
            for i in range(3):
                x[bead_index, i] = nmc[i, bead_index]
        else:
            try:
                p = PyCapsule_GetPointer(self.plans[(FFTW_BACKWARD, nb)], PLAN_CAPSULE_NAME)
            except KeyError:
                p = fftw_plan_dft_1d(nb, <fftw_complex *>w1, <fftw_complex *>w2,
                                     FFTW_BACKWARD, FFTW_ESTIMATE)
                self.plans[(FFTW_BACKWARD, nb)] = \
                        PyCapsule_New(p, PLAN_CAPSULE_NAME, plan_capsule_destructor)
            for i in range(3):
                w1[0] = nmc[i, bead_index+0]
                w1[1] = 0.
                for j in range(1, (nb+1)/2):
                    w1[2*j] = nmc[i, bead_index+j]
                    w1[2*j+1] = nmc[i, bead_index+nb-j]
                    w1[2*nb-2*j] = w1[2*j]
                    w1[2*nb-2*j+1] = -w1[2*j+1]
                if nb % 2 == 0:
                    w1[nb] = nmc[i, bead_index+nb/2]
                    w1[nb+1] = 0.
                fftw_execute(p)
                for j in range(nb):
                    x[bead_index+j, i] = w2[2*j]/nb

    cdef void propagateOscillators(self, N.ndarray[double, ndim=2] nmc,
                                   N.ndarray[double, ndim=2] nmv,
                                   Py_ssize_t bead_index, int32_t nb, double beta, double dt):
        cdef double omega_n = nb/(beta*hbar)
        cdef double omega_k, omega_k_dt, s, c
        cdef double temp
        cdef Py_ssize_t i, k
        for i in range(3):
            nmc[i, bead_index] += dt*nmv[i, bead_index]
            for k in range(1, nb):
                omega_k = 2.*omega_n*sin(k*M_PI/nb)
                omega_k_dt = omega_k*dt
                s = sin(omega_k_dt)
                c = cos(omega_k_dt)
                temp = c*nmv[i, bead_index+k]-omega_k*s*nmc[i, bead_index+k]
                nmc[i, bead_index+k] = s*nmv[i, bead_index+k]/omega_k + c*nmc[i, bead_index+k]
                nmv[i, bead_index+k] = temp

    cdef double springEnergyNormalModes(self, N.ndarray[double, ndim=2] nmc,
                                        N.ndarray[double, ndim=1] m,
                                        N.ndarray[N.int32_t, ndim=2] bd,
                                        double beta):
        cdef Py_ssize_t i, j, k
        cdef int32_t nb
        cdef double sumsq
        cdef double omega_n, omega_k
        cdef double e = 0.
        for i in range(nmc.shape[1]):
            if bd[i, 0] == 0:
                nb = bd[i, 1]
                omega_n = nb/(beta*hbar)
                # Start at j=1 because the contribution from the centroid is zero
                for j in range(1, nb):
                    omega_k = 2.*omega_n*sin(j*M_PI/nb)
                    sumsq = 0.
                    for k in range(3):
                        sumsq += nmc[k, i+j]*nmc[k, i+j]
                    # j=nb/2 corresponds to the real-valued coordinate at
                    # the maximal frequency.
                    if nb % 2 == 0 and j == nb/2:
                        sumsq *= 0.5
                    e += m[i]*sumsq*omega_k*omega_k/nb
        return e

    cdef void applyThermostat(self, N.ndarray[double, ndim=2] v, N.ndarray[double, ndim=2] nmv,
                              N.ndarray[double, ndim=1] m, N.ndarray[N.int32_t, ndim=2] bd,
                              double dt, double beta):
        pass

    cdef void atomtocm(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] v,
                       N.ndarray[double, ndim=2] g, N.ndarray[double, ndim=1] m,
                       N.ndarray[double, ndim=2] xcm, N.ndarray[double, ndim=2] vcm,
                       N.ndarray[double, ndim=2] gcm, N.ndarray[double, ndim=1] mcm,
                       N.ndarray[N.int32_t, ndim=2] bdcm, int Nmol):

         cdef int tot_atoms,i,j,k,z,natomspmol,nbeadspmol, atom_index
         tot_atoms=0
         for i in range (Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            # nbeadspmol is the number of beads we want our molecule COM to have. 
            # Therefore is the number of beads each atom has in the molecule.
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol          
            
            for z in range (nbeadspmol):
                bdcm[i*nbeadspmol+z,0]=N.int32(z)
                if bdcm[i*nbeadspmol+z,0] == N.int32(0):
                    bdcm[i*nbeadspmol+z,1]=N.int32(nbeadspmol)
                mcm[i*nbeadspmol+z]=self.universe.objectList()[i].mass()/nbeadspmol
                
                
                for k in range(3):
                    xcm[i*nbeadspmol+z,k]=0.0
                    vcm[i*nbeadspmol+z,k]=0.0
                    gcm[i*nbeadspmol+z,k]=0.0
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        xcm[i*nbeadspmol+z,k]+=m[atom_index*nbeadspmol+z]*x[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                        vcm[i*nbeadspmol+z,k]+=m[atom_index*nbeadspmol+z]*v[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                        gcm[i*nbeadspmol+z,k]+=g[atom_index*nbeadspmol+z,k]

            tot_atoms+=natomspmol

    cdef void cmtoatom(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] v,
                       N.ndarray[double, ndim=2] g, N.ndarray[double, ndim=1] m,
                       N.ndarray[double, ndim=2] xcm, N.ndarray[double, ndim=2] vcm,
                       N.ndarray[double, ndim=2] gcm, N.ndarray[double, ndim=1] mcm,
                       int Nmol):

         #xcom is ORIGINAL center of mass!
         cdef N.ndarray[double,ndim=1] xcom
         cdef int tot_atoms,i,j,k,z,natomspmol,nbeadspmol, atom_index

         xcom=N.zeros((3,),N.float)

         tot_atoms=0

         for i in range (Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol
            for z in range (nbeadspmol):
                for k in range(3):
                    xcom[k]=0.
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        xcom[k]+=m[atom_index*nbeadspmol+z]*x[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                for k in range(3):
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        x[atom_index*nbeadspmol+z,k]=x[atom_index*nbeadspmol+z,k]-xcom[k]+xcm[i*nbeadspmol+z,k]
                        g[atom_index*nbeadspmol+z,k]=gcm[i*nbeadspmol+z,k]*m[atom_index*nbeadspmol+z]/mcm[i*nbeadspmol+z]
                        v[atom_index*nbeadspmol+z,k]=vcm[i*nbeadspmol+z,k]

            tot_atoms+=natomspmol

    cdef void eulertocart(self, int bindex, int molnum, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=1] eulerangles, N.ndarray[double, ndim=1] bondlength, N.ndarray[double,ndim=2] xcm):

        natomspmol=self.universe.objectList()[molnum].numberOfAtoms()
        nbeadspmol=self.universe.objectList()[molnum].numberOfPoints()/natomspmol
        for j in range(natomspmol):
            aindex=bindex+nbeadspmol*j
            v1=bondlength[aindex]*Vector(eulerangles[0],eulerangles[1],eulerangles[2])
            for i in range(3):
                x[aindex,i]=xcm[molnum*nbeadspmol+(bindex%nbeadspmol),i]+v1[i]

    def energyCalculator(self, x):
        cdef energy_data energytemp
        energytemp.gradients = NULL
        energytemp.gradient_fn = NULL
        energytemp.force_constants = NULL
        energytemp.fc_fn = NULL
        self.calculateEnergies(x, &energytemp, 0)
        return energytemp.energy

    cdef start(self):
        #cdef clock_t timep0, timep1, timetransstart, timerotstart, timeinitstart, timepot1start, timeanalysis1start
        #cdef clock_t timeconvertstart, timepot2start, timeanalysis2start, timeanalysis2end, timerotend
        #cdef double timeinit, timepot1, timeanalysis1, timeconvert, timepot2, timeanalysis2, timeroteng

        #cdef double time0, timerot, timetrans
        cdef double acceptratio, rd, sint, pot_old, pot_new, dens_old, dens_new, indexp0val, indexp1val
        cdef int t0b,t1b,t2b,t0,t1,t2,atombead,indexp0,indexp1,indexp0n,indexp1n

        cdef N.ndarray[double, ndim=2] x, v, g, dv, nmc, nmv, xcm, vcm, gcm
        cdef N.ndarray[double, ndim=1] m, mcm
        cdef N.ndarray[double, ndim=1] bondlength
        cdef N.ndarray[N.int32_t, ndim=2] bd, bdcm
        cdef N.ndarray[double, ndim=3] ss
        cdef energy_data energy
        cdef double time, delta_t, ke, ke_nm, se, beta, temperature
        cdef double qe_prim, qe_vir, qe_cvir, qe_rot
        cdef int natoms, nbeads, nsteps, step, df, cdf, nb, Nmol, Ntruemol,rotbdcount,rotbdskip
        cdef Py_ssize_t i, j, k

        cdef double propct, propphi
        cdef int P
        cdef N.ndarray[double, ndim=1] costheta,phi
        cdef N.ndarray[double, ndim=2] MCCosine
        cdef N.ndarray[double, ndim=1] MCCosprop
        cdef N.ndarray[double, ndim=2] xold
        cdef N.ndarray[double, ndim=1] densitymatrix, rotenergy
        cdef double rotstep,ndens
        cdef int rotskipstep, nrotsteps
        densitymatrix=self.densmat
        ndens=1.0*len(densitymatrix)
        rotenergy=self.rotengmat
        rotstep=self.rotmove
        rotskipstep=self.rotstepskip

        #timep0=clock()
        #timetrans=0.0
        #timerot=0.0
        #timeinit=0.0
        #timepot1=0.0
        #timeanalysis1=0.0
        #timeconvert=0.0
        #timepot2=0.0
        #timeanalysis2=0.0
        #timeroteng=0.0

        # Check if velocities have been initialized
        if self.universe.velocities() is None:
            raise ValueError("no velocities")

        # Gather state variables and parameters
        configuration = self.universe.configuration()
        velocities = self.universe.velocities()
        gradients = ParticleProperties.ParticleVector(self.universe)
        masses = self.universe.masses()
        delta_t = self.getOption('delta_t')
        nsteps = self.getOption('steps')
        natoms = self.universe.numberOfAtoms()
        nbeads = self.universe.numberOfPoints()
        bd = self.evaluator_object.global_data.get('bead_data')
        pi_environment = self.universe.environmentObjectList(Environment.PathIntegrals)[0]
        beta = pi_environment.beta

        # For efficiency, the Cython code works at the array
        # level rather than at the ParticleProperty level.
        x = configuration.array
        v = velocities.array
        g = gradients.array
        m = masses.array

	# MATT-Introduce X-COM variable, number of molecules Nmol
        acceptratio=0.0
        P=nbeads/natoms
        Nmol = len(self.universe.objectList())
        nbeads_mol = N.int32(P*Nmol)
        xcm = N.zeros((nbeads_mol, 3), N.float)
        vcm = N.zeros((nbeads_mol, 3), N.float)
        gcm = N.zeros((nbeads_mol, 3), N.float)
        mcm = N.zeros(nbeads_mol, N.float)
        dv = N.zeros((nbeads_mol, 3), N.float)
        nmc = N.zeros((3, nbeads_mol), N.float)
        nmv = N.zeros((3, nbeads_mol), N.float)
        bdcm = N.zeros((nbeads_mol,2), N.int32)
        bondlength=N.zeros(nbeads,N.float)

        #ROTATIONAL VARIABLES
        nrotsteps=0
        costheta = N.zeros(nbeads_mol, N.float)
        phi = N.zeros(nbeads_mol, N.float)
        MCCosine = N.zeros((nbeads_mol,3), N.float)
        MCCosprop=N.zeros(3, N.float)

        # Check if there is a frozen_subspace
        subspace = self.getOption('frozen_subspace')
        if subspace is None:
            ss = N.zeros((0, nbeads_mol, 3), N.float)
            df = 3*nbeads_mol
            cdf = 3*Nmol
        else:
            ss = subspace.getBasis().array
            df = 3*nbeads-ss.shape[0]
            cdf = self.centroidDegreesOfFreedom(subspace, bdcm)

        # Initialize the plan cache.
        self.plans = {}

        # Ask for energy gradients to be calculated and stored in
        # the array g. Force constants are not requested.
        energy.gradients = <void *>g
        energy.gradient_fn = NULL
        energy.force_constants = NULL
        energy.fc_fn = NULL

        # Declare the variables accessible to trajectory actions.
        self.declareTrajectoryVariable_double(
            &time, "time", "Time: %lf\n", time_unit_name, PyTrajectory_Time)
        self.declareTrajectoryVariable_array(
            v, "velocities", "Velocities:\n", velocity_unit_name,
            PyTrajectory_Velocities)
        self.declareTrajectoryVariable_array(
            g, "gradients", "Energy gradients:\n", energy_gradient_unit_name,
            PyTrajectory_Gradients)
        self.declareTrajectoryVariable_double(
            &energy.energy,"potential_energy", "Potential energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &ke, "kinetic_energy", "Kinetic energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &se, "spring_energy", "Spring energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &temperature, "temperature", "Temperature: %lf\n",
            temperature_unit_name, PyTrajectory_Thermodynamic)
        self.declareTrajectoryVariable_double(
            &qe_prim, "quantum_energy_primitive",
            "Primitive quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_vir, "quantum_energy_virial",
            "Virial quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_cvir, "quantum_energy_centroid_virial",
            "Centroid virial quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_rot, "quantum_energy_rotation",
            "Rotation quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.initializeTrajectoryActions()

        # Acquire the write lock of the universe. This is necessary to
        # make sure that the integrator's modifications to positions
        # and velocities are synchronized with other threads that
        # attempt to use or modify these same values.
        #
        # Note that the write lock will be released temporarily
        # for trajectory actions. It will also be converted to
        # a read lock temporarily for energy evaluation. This
        # is taken care of automatically by the respective methods
        # of class EnergyBasedTrajectoryGenerator.
        self.acquireWriteLock()

        # Preparation: Calculate initial half-step accelerations
        # and run the trajectory actions on the initial state.
        self.foldCoordinatesIntoBox()

        Ntruemol=0
        for i in range(Nmol):
            print i, self.universe.objectList()[i].numberOfAtoms()
            if (self.universe.objectList()[i].numberOfAtoms()>1):
                Ntruemol+=1
        print Ntruemol
                

        for i in range (Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            # nbeadspmol is the number of beads we want our molecule COM to have.
            # Therefore is the number of beads each atom has in the molecule.
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol

            for z in range (nbeadspmol):
                    bdcm[i*nbeadspmol+z,0]=N.int32(z)
                    if bdcm[i*nbeadspmol+z,0] == N.int32(0):
                        bdcm[i*nbeadspmol+z,1]=N.int32(nbeadspmol)
                    mcm[i*nbeadspmol+z]=self.universe.objectList()[i].mass()/nbeadspmol
										

	
        # Allocate workspace for Fourier transforms
        nb_max = bdcm[:, 1].max()
        self.workspace1 = N.zeros((2*nb_max,), N.float)
        self.workspace_ptr_1 = <double *>self.workspace1.data
        self.workspace2 = N.zeros((2*nb_max,), N.float)
        self.workspace_ptr_2 = <double *>self.workspace2.data


        #Calculate Energy and Fill Gradient Vector
        self.calculateEnergies(x, &energy, 0)
        self.atomtocm(x,v,g,m,xcm,vcm,gcm,mcm,bdcm,Nmol)


        ##########################################
        ### CALCULATE ANGLES AND FILL MCCosine ###
        ##########################################
        atomcount=-1
        for i in range(Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol
            atomstart=atomcount+1
            atomend=atomcount+natomspmol
            for j in range(natomspmol):
                atomcount+=1
                if (natomspmol>1):
                   for p in range(nbeadspmol):
                       bondlength[atomcount*nbeadspmol+p]=N.dot((N.asarray(x[atomcount*nbeadspmol+p])-xcm[i*nbeadspmol+p]),(N.asarray(x[atomend*nbeadspmol+p]-x[atomstart*nbeadspmol+p])))/N.linalg.norm(x[atomend*nbeadspmol+p]-x[atomstart*nbeadspmol+p])


        atomcount=-1
        for k in range(Nmol):
            natomspmol=self.universe.objectList()[k].numberOfAtoms()
            nbeadspmol=self.universe.objectList()[k].numberOfPoints()/natomspmol
            atomcount+=natomspmol
            if (natomspmol>1):
                for i in range(nbeadspmol):
                    rel=x[atomcount*nbeadspmol+p]-x[(atomcount-1)*nbeadspmol+p]
                    costheta[k*nbeadspmol+i]=N.dot(N.asarray(rel), N.asarray([0.,0.,1.]))/N.linalg.norm(N.asarray(rel))
                    if (abs(rel[0])<1.0e-16):
                        if (abs(rel[1])<1.0e-16):
                            phi[k*nbeadspmol+i]=0.0
                        elif (N.sign(rel[0])==N.sign(rel[1])):
                            phi[k*nbeadspmol+i]=N.pi/2.0
                        else:
                            phi[k*nbeadspmol+i]=-1.0*N.pi/2.0
                    else:
                        phi[k*nbeadspmol+i]=N.arctan(rel[1]/rel[0])
                    sint=sqrt(1.0-costheta[k*nbeadspmol+i]*costheta[k*nbeadspmol+i])
                    MCCosine[k*nbeadspmol+i][0]=sint*N.cos(phi[k*nbeadspmol+i])
                    MCCosine[k*nbeadspmol+i][1]=sint*N.sin(phi[k*nbeadspmol+i])
                    MCCosine[k*nbeadspmol+i][2]=costheta[k*nbeadspmol+i]

        #print "Atom to CM"
        #print g
        #print gcm

        self.freeze(vcm, ss)

        for i in range(nbeads_mol):
            if bdcm[i, 0] == 0:
                self.fixBeadPositions(xcm, i, bdcm[i, 1])
                self.cartesianToNormalMode(xcm, nmc, i, bdcm[i, 1])

        se = self.springEnergyNormalModes(nmc, mcm, bdcm, beta)
        qe_prim = energy.energy - se + 0.5*df/beta
        #qe_vir = energy.energy - 0.5*energy.virial
        #qe_cvir = energy.energy \
        #          - 0.5*self.centroidVirial(x, g, bd) \
        #          + 0.5*cdf/beta

        ke = 0.
        for i in range(nbeads_mol):
            for j in range(3):
                dv[i, j] = -0.5*delta_t*gcm[i, j]/mcm[i]
                ke += 0.5*mcm[i]*vcm[i, j]*vcm[i, j]
        temperature = 2.*ke/(df*k_B)

        #print "Before check FFT"
        #print g
        #print gcm

        # Check FFT
        if False:
            xcm_test = N.zeros((nbeads_mol, 3), N.float)
            vcm_test = N.zeros((nbeads_mol, 3), N.float)
            for i in range(nbeads_mol):
                if bdcm[i, 0] == 0:
                    self.cartesianToNormalMode(xcm, nmc, i, bdcm[i, 1])
                    self.normalModeToCartesian(xcm_test, nmc, i, bdcm[i, 1])
                    self.cartesianToNormalMode(vcm, nmv, i, bdcm[i, 1])
                    self.normalModeToCartesian(vcm_test, nmv, i, bdcm[i, 1])
            for i in range(nbeads_mol):
                for j in range(3):
                    assert fabs(xcm[i, j]-xcm_test[i, j]) < 1.e-7
                    assert fabs(vcm[i, j]-vcm_test[i, j]) < 1.e-7

        #timep1=clock()
        #time0 = (<double> (timep1 - timep0)) / CLOCKS_PER_SEC
        #print "Initialization Time (s): ", time0
        # Main integration loop
        time = 0.

        self.trajectoryActions(0)

        #print "Before integration step"
        #print g
        #print gcm

        for step in range(nsteps):
            #timetransstart=clock()
	    
            # First application of thermostat
            self.applyThermostat(vcm, nmv, mcm, bdcm, delta_t, beta)
            # First integration half-step
            for i in range(nbeads_mol):
                for j in range(3):
                    dv[i, j] = -0.5*delta_t*gcm[i, j]/mcm[i]
                    vcm[i, j] += dv[i, j]
            # Remove frozen subspace
            self.freeze(vcm, ss)

            #print "After Apply Thermostat"
            #print g
            #print gcm

            # Conversion to normal mode coordinates
            for i in range(nbeads_mol):
                # bd[i, 0] == 0 means "first bead of an atom"
                if bdcm[i, 0] == 0:
                    self.fixBeadPositions(xcm, i, bdcm[i, 1])
                    self.cartesianToNormalMode(xcm, nmc, i, bdcm[i, 1])
                    self.cartesianToNormalMode(vcm, nmv, i, bdcm[i, 1])

            # Harmonic oscillator time propagation
            for i in range(nbeads_mol):
                # bd[i, 0] == 0 means "first bead of an atom"
                if bdcm[i, 0] == 0:
                    self.propagateOscillators(nmc, nmv, i, bdcm[i, 1], beta, delta_t)
            # Conversion back to Cartesian coordinates
            for i in range(nbeads_mol):
                # bd[i, 0] == 0 means "first bead of an atom"
                if bdcm[i, 0] == 0:
                    self.normalModeToCartesian(xcm, nmc, i, bdcm[i, 1])
                    self.normalModeToCartesian(vcm, nmv, i, bdcm[i, 1])


            # Mid-step energy calculation
            self.cmtoatom(x,v,g,m,xcm,vcm,gcm,mcm,Nmol)

            self.calculateEnergies(x, &energy, 1)
            self.atomtocm(x,v,g,m,xcm,vcm,gcm,mcm,bdcm,Nmol)

            #print "After Energy Calculation"
            #print g
            #print gcm

            # Quantum energy estimators
            se = self.springEnergyNormalModes(nmc, mcm, bdcm, beta)

            qe_prim = energy.energy - se + 0.5*df/beta
            #qe_vir = energy.energy - 0.5*energy.virial
            #qe_cvir = energy.energy \
            #          - 0.5*self.centroidVirial(x, g, bd) \
            #          + 0.5*cdf/beta

            # Second integration half-step
            for i in range(nbeads_mol):
                for j in range(3):
                    dv[i, j] = -0.5*delta_t*gcm[i, j]/mcm[i]
                    vcm[i, j] += dv[i, j]
            # Second application of thermostat

            #print "After Conversion to CM"
            #print g
            #print gcm

            self.applyThermostat(vcm, nmv, mcm, bdcm, delta_t, beta)

            # Remove frozen subspace
            self.freeze(vcm, ss)

            #CHECK
            for i in range(nbeads_mol):
                if bdcm[i, 0] == 0:
                    self.cartesianToNormalMode(vcm, nmv, i, bdcm[i, 1])


            # Calculate kinetic energy
            ke = 0.
            for i in range(nbeads_mol):
                for j in range(3):
                    ke += 0.5*mcm[i]*vcm[i, j]*vcm[i, j]
            temperature = 2.*ke/(df*k_B)
            if False:
                ke_nm = 0.
                for i in range(nbeads_mol):
                    if bdcm[i, 0] == 0:
                        for j in range(3):
                            for k in range(bdcm[i,1]):
                                if k == 0 or (bdcm[i,1] % 2 == 0 and k == bdcm[i,1]/2):
                                    ke_nm += 0.5*mcm[i]*nmv[j, i+k]*nmv[j, i+k]/bdcm[i,1]
                                else:
                                    ke_nm += mcm[i]*nmv[j, i+k]*nmv[j, i+k]/bdcm[i,1]
                assert fabs(ke-ke_nm) < 1.e-7

            self.cmtoatom(x,v,g,m,xcm,vcm,gcm,mcm,Nmol)
            pot_old=energy.energy

            #print "After Translation"
            #print g
            #print gcm

            #timerotstart=clock()
            #timetrans+=(<double> (timerotstart - timetransstart)) / CLOCKS_PER_SEC

            #######################################
            ### PERFORM MC RIGID BODY ROTATIONS ###
            #######################################

            if (step%rotskipstep == 0):
                nrotsteps+=1
                rotbdcount=1
                rotbdskip=1
                for stp in range(rotbdcount):
                    for t1b in range(stp%rotbdskip,P,rotbdskip):
                        atomcount=0
                        for a in range(Nmol):
                            natomspmol=self.universe.objectList()[a].numberOfAtoms()
    
                            if (natomspmol==1):
                                atomcount+=natomspmol
                                continue
    
                            #timeinitstart=clock()
                            t0b=t1b-1
                            t2b=t1b+1
                     
                            if (t0b<0): t0b+=P
                            if (t2b>(P-1)): t2b-=P
    
                            t0=a*P+t0b
                            t1=a*P+t1b
                            t2=a*P+t2b
    	         
                            atombead=atomcount*P+t1b
                            xold=N.zeros((natomspmol,3),float)
                            for i in range(natomspmol):
                                for j in range(3):
                                    xold[i,j]=x[atombead+i*P,j]
                            propct=costheta[t1]+rotstep*(N.random.random()-0.5)
                            propphi=phi[t1]+rotstep*(N.random.random()-0.5)
    	                
                            if (propct > 1.0):
                                propct=2.0-propct
                            elif (propct < -1.0):
                                propct=-2.0-propct
    	         
                            sint=sqrt(1.0-propct*propct)
    	                
                            MCCosprop[0]=sint*N.cos(propphi)
                            MCCosprop[1]=sint*N.sin(propphi)
                            MCCosprop[2]=propct
    	         
                            #timepot1start=clock()
                            #timeanalysis1start=clock()
                            p0=0.0
                            p1=0.0
                            for co in range(3):
                                p0+=MCCosine[t0][co]*MCCosine[t1][co]
                                p1+=MCCosine[t1][co]*MCCosine[t2][co]
    	         
    	         
                            indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))
                            indexp1=int(N.floor((p1+1.0)*(ndens-1.0)/2.0))
    	         
                            indexp0n=indexp0+1
                            indexp1n=indexp1+1
                            if (indexp0==ndens-1):
                                indexp0n=indexp0
                            if (indexp1==ndens-1):
                                indexp1n=indexp1
    	         
                            indexp0val=-1.0+indexp0*2.0/(ndens-1.0)
                            indexp1val=-1.0+indexp1*2.0/(ndens-1.0)
    	         
                            dens_old=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))
    	         
    
                            if (fabs(dens_old)<(1.0e-10)):
                                dens_old=0.0
                            if (dens_old < 0.0):
                                print "Rotational Density Negative"
                                raise()
    
    	                
                            ##NEW DENSITY
                            p0=0.0
                            p1=0.0
                            for co in range(3):
                                p0+=MCCosine[t0][co]*MCCosprop[co]
                                p1+=MCCosprop[co]*MCCosine[t2][co]
    	         
                            indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))
                            indexp1=int(N.floor((p1+1.0)*(ndens-1.0)/2.0))
    	         
                            indexp0n=indexp0+1
                            indexp1n=indexp1+1
                            if (indexp0==ndens-1):
                                indexp0n=indexp0
                            if (indexp1==ndens-1):
                                indexp1n=indexp1
    	         
                            indexp0val=-1.0+indexp0*2.0/(ndens-1.0)
                            indexp1val=-1.0+indexp1*2.0/(ndens-1.0)
    	         
                            dens_new=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))
    	         
                            if (fabs(dens_new)<(1.0e-10)):
                                dens_new=0.0
                            if (dens_new < 0.0):
                                print "Rotational Density Negative"
                                raise()
    
                            #timeconvertstart=clock()
                            self.eulertocart(atombead, a,x, MCCosprop, bondlength,xcm)
    
                            #timepot2start=clock()
                            pot_new=self.energyCalculator(N.asarray(x))
    
                            #timeanalysis2start=clock()
                            rd=1.0
                            if (dens_old>(1.0e-10)):
                                rd=dens_new/dens_old
    	         
                            rd*= exp(-(beta/P)*(pot_new-pot_old))
    	         
                            accept=False
                            if (rd>1.0):
                                accept=True
                            elif (rd>N.random.random()):
                                accept=True
    	                
                            if (accept):
                                pot_old=pot_new
                                acceptratio+=1.0
                                costheta[t1]=propct
                                phi[t1]=propphi
                                for co in range(3):
                                    MCCosine[t1][co]=MCCosprop[co]
                            else:
                                for i in range(natomspmol):
                                    for j in range(3):
                                        x[atombead+i*P,j]=xold[i,j]
                            atomcount+=natomspmol
                            #timeanalysis2end=clock()
                            #timeinit+=(<double> (timepot1start - timeinitstart)) / CLOCKS_PER_SEC
                            #timepot1+=(<double> (timeanalysis1start - timepot1start)) / CLOCKS_PER_SEC
                            #timeanalysis1+=(<double> (timeconvertstart - timeanalysis1start)) / CLOCKS_PER_SEC
                            #timeconvert+=(<double> (timepot2start - timeconvertstart)) / CLOCKS_PER_SEC
                            #timepot2+=(<double> (timeanalysis2start - timepot2start)) / CLOCKS_PER_SEC
                            #timeanalysis2+=(<double> (timeanalysis2end - timeanalysis2start)) / CLOCKS_PER_SEC
            qe_rot=0.0
            for a in range(Nmol):
                if (self.universe.objectList()[a].numberOfAtoms() == 1):
                    continue
                for t1b in range(P):
                    t0b=t1b-1
                    if (t0b<0): t0b+=P

                    t0=a*P+t0b
                    t1=a*P+t1b
                    p0=0.0
                    for co in range(3):
                        p0+=MCCosine[t0][co]*MCCosine[t1][co]

                    indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))

                    indexp0n=indexp0+1
                    if (indexp0==ndens-1):
                        indexp0n=indexp0

                    indexp0val=-1.0+indexp0*2.0/(ndens-1.0)

                    qe_rot+=rotenergy[indexp0]+(rotenergy[indexp0n]-rotenergy[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0))

            self.calculateEnergies(x, &energy, 0)
            self.atomtocm(x,v,g,m,xcm,vcm,gcm,mcm,bdcm,Nmol)

            #timerotend=clock()
            #timeroteng+=(<double> (timerotend - timeanalysis2end)) / CLOCKS_PER_SEC

            #print "After Rotation"
            #print g
            #print gcm

            # End of time step
            time += delta_t
            self.foldCoordinatesIntoBox()
            self.trajectoryActions(step+1)

        # Release the write lock.
        self.releaseWriteLock()

        # Finalize all trajectory actions (close files etc.)
        self.finalizeTrajectoryActions(nsteps)

        # Deallocate the Fourier transform workspace
        self.workspace_ptr_1 = NULL
        self.workspace_ptr_2 = NULL
        self.workspace1 = None
        self.workspace2 = None

        #Calc Timing
        #timetrans/=nsteps
        #timeinit/=nsteps
        #timepot1/=nsteps
        #timeanalysis1/=nsteps
        #timeconvert/=nsteps
        #timepot2/=nsteps
        #timeanalysis2/=nsteps
        #timeroteng/=nsteps
        #print "Avg Time for Translation (s): ",timetrans
        #print "Avg Time for Rot Init (s): ",timeinit
        #print "Avg Time for Calc Pot1 (s): ",timepot1
        #print "Avg Time for Analysis 1 (s): ",timeanalysis1
        #print "Avg Time for Convert (s): ",timeconvert
        #print "Avg Time for Calc Pot 2 (s): ",timepot2
        #print "Avg Time for Analysis 2 (s): ",timeanalysis2
        #print "Avg Time for Rot Eng (s): ",timeroteng

        acceptratio/=Ntruemol*float(P*nrotsteps*rotbdcount/rotbdskip)
        print "Acceptance Ratio: ", acceptratio


#
# Velocity Verlet integrator in normal-mode coordinates
# with a Langevin thermostat
#
cdef class RigidRotor_PILangevinNormalModeIntegrator(RigidRotor_PINormalModeIntegrator):

    """
    Molecular dynamics integrator for path integral systems using
    normal-mode coordinates and a Langevin thermostat.

    This integrator works like PINormalModeIntegrator, but has
    an additional option "centroid_friction", which is a ParticleScalar
    (one friction constant per atom) or a plain number.

    """

    cdef N.ndarray gamma
    
    cdef void applyThermostat(self, N.ndarray[double, ndim=2] v, N.ndarray[double, ndim=2] nmv,
                              N.ndarray[double, ndim=1] m, N.ndarray[N.int32_t, ndim=2] bd,
                              double dt, double beta):
        cdef N.ndarray[double, ndim=1] g = self.gamma
        cdef int nbeads = v.shape[0]
        cdef double f, c1, c2
        cdef double omega_n, mb
        cdef Py_ssize_t i, j, k
        cdef int32_t nb
        for i in range(nbeads):
            # bd[i, 0] == 0 means "first bead of an atom"
            if bd[i, 0] == 0:
                nb = bd[i, 1]
                # Conversion to normal mode coordinates
                self.cartesianToNormalMode(v, nmv, i, nb)
                # Modify velocities
                omega_n = nb/(beta*hbar)
                mb = sqrt(nb/(beta*m[i]))
                for k in range(nb):
                    if k == 0:
                        f = g[i]
                    else:
                        f = 4.*omega_n*sin(k*M_PI/nb)
                    c1 = exp(-0.5*dt*f)
                    c2 = sqrt(1-c1*c1)
                    for j in range(3):
                        if k == 0 or (nb % 2 == 0 and k == nb/2):
                            nmv[j, i+k] = c1*nmv[j, i+k] + c2*mb*MMTK.mtrand.standard_normal()
                        else:
                            nmv[j, i+k] = c1*nmv[j, i+k] + sqrt(0.5)*c2*mb*MMTK.mtrand.standard_normal()
                # Conversion back to Cartesian coordinates
                self.normalModeToCartesian(v, nmv, i, nb)

    cdef start(self):
        friction = self.getOption('centroid_friction')
        self.densmat=self.getOption('densmat')
        self.rotengmat=self.getOption('rotengmat')
        self.rotmove=self.getOption('rotstep')
        self.rotstepskip=self.getOption('rotskipstep')
        if isinstance(friction, ParticleProperties.ParticleScalar):
            self.gamma = friction.array
        else:
            assert isinstance(friction, numbers.Number)
            nbeads = self.universe.numberOfPoints()
            self.gamma = N.zeros((nbeads,), N.float)+friction
        RigidRotor_PINormalModeIntegrator.start(self)

