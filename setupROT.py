#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext

from distutils.core import setup, Command, Extension
from distutils.command.build import build
from distutils.command.sdist import sdist
from distutils.command.install_data import install_data
from distutils import dir_util
from distutils.filelist import FileList, translate_pattern
import distutils.sysconfig
sysconfig = distutils.sysconfig.get_config_vars()


import os, sys, types
import ctypes, ctypes.util
from glob import glob

#os.environ["CC"] = "g++" 
#os.environ["CXX"] = "g++"

# Check for Cython and use it if the environment variable
# MMTK_USE_CYTHON is set to a non-zero value.
use_cython = int(os.environ.get('MMTK_USE_CYTHON', '0')) != 0
if use_cython:
    try:
        from Cython.Build import cythonize
        use_cython = True
    except ImportError:
        use_cython = False


compile_args = []
prefix_user='/home/mdgschmi/'
include_dirs= [prefix_user+'Dev4/include',prefix_user+'Include']
libraries = []
macros = []

from Scientific import N
try:
    num_package = N.package
except AttributeError:
    num_package = "Numeric"
if num_package == "NumPy":
    compile_args.append("-DNUMPY=1")
    import numpy.distutils.misc_util
    include_dirs.extend(numpy.distutils.misc_util.get_numpy_include_dirs())

headers = glob(os.path.join("Include", "MMTK", "*.h"))
headers.extend(glob(os.path.join("Include", "MMTK", "*.px[di]")))

#################################################################
# Check various compiler/library properties
library_dirs = ['/home/mdgschmi/Dev4/lib']

libraries = []
if sysconfig['LIBM'] != '':
    libraries.append('m')

macros = []
try:
    from Scientific.MPI import world
except ImportError:
    world = None
if world is not None:
    if type(world) == types.InstanceType:
        world = None
if world is not None:
    macros.append(('WITH_MPI', None))

if hasattr(ctypes.CDLL(ctypes.util.find_library('m')), 'erfc'):
    macros.append(('LIBM_HAS_ERFC', None))

if sys.platform != 'win32':
    if ctypes.sizeof(ctypes.c_long) == 8:
        macros.append(('_LONG64_', None))

if sys.version_info[0] == 2 and sys.version_info[1] >= 2:
    macros.append(('EXTENDED_TYPES', None))


#################################################################
# System-specific optimization options

low_opt = []
if sys.platform != 'win32' and 'gcc' in sysconfig['CC']:
    low_opt = ['-O0']
low_opt.append('-g')

high_opt = []
if sys.platform[:5] == 'linux' and 'gcc' in sysconfig['CC']:
    high_opt = ['-O3', '-ffast-math', '-fomit-frame-pointer',
                '-fkeep-inline-functions']
if sys.platform == 'darwin' and 'gcc' in sysconfig['CC']:
    high_opt = ['-O3', '-ffast-math', '-fomit-frame-pointer',
                '-fkeep-inline-functions']
if sys.platform == 'aix4':
    high_opt = ['-O4']
if sys.platform == 'odf1V4':
    high_opt = ['-O2', '-fp_reorder', '-ansi_alias', '-ansi_args']

high_opt.append('-g')

ext_modules=[
                      Extension('MMTK.RigidRotor_PINormalModeIntegrator_test',
                                ['Src/MMTK/RigidRotor_PINormalModeIntegrator_test.pyx'],
                                extra_compile_args = compile_args,
                                include_dirs=include_dirs,
                                library_dirs=library_dirs,
                                libraries=libraries+ ['fftw3'],
                                define_macros=macros)
]

setup(
  name = 'Demos',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
