# RigidLinearRotor-PIMCD
This code performs rigid body rotations and translations for linear molecules. Rotation sampling is done using Metropolis Path Integral Monte Carlo and translation sampling is done using rejection-free Path Integral Molecular Dynamics using the Langevin equation thermostat. The Molecular Modelling ToolKit (MMTK) is required for use.

Note: This is currently undergoing testing and debugging.

One will be required to download the cython RigidRotor_PINormalModeIntegrator_test.pyx and compile it using a setup.py file that is familiar to the Roy Group.
