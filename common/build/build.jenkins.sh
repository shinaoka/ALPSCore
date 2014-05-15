# Script used by the build server the build
# all modules of ALPSCore

# This script expects the following environment variables
# BOOST_ROOT - location for boost distribution
# HDF5_ROOT - location for the HDF5 distribution
# MPI_CXX_COMPILER - location for the mpi c++ compile

# Make sure we are in top directory for the repository
SCRIPTDIR=$(dirname $0)
cd $SCRIPTDIR/../..

# Function to build one module
function build {
  MODULE=$1
  MODULEDIR=$PWD/$MODULE
  INSTALLDIR=$PWD/install
  BUILDDIR=$PWD/build.tmp

  rm -rf $BUILDDIR
  mkdir $BUILDDIR
  cd $BUILDDIR

  echo "*** Building module in $MODULEDIR to $INSTALLDIR ***"

  cmake \
  -DCMAKE_INSTALL_PREFIX="${TARGETDIR}" \
  -DTesting=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  ${MODULEDIR}

  make || exit 1 
  make test || exit 1
  make install || exit 1

  cd ..
}

build utility
build hdf5

echo "*** Done ***"