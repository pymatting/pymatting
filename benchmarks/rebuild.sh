BUILD_DIR=$(python -c "import config; print(config.BUILD_DIR)")
rm -rf $BUILD_DIR/pymatting/
mkdir -p $BUILD_DIR/pymatting/
cp *.c $BUILD_DIR/pymatting/
cp *.cpp $BUILD_DIR/pymatting/
cd $BUILD_DIR/pymatting/

echo "building mumps"
mpicc solve_mumps.c -O3 -fPIC -shared -ldmumps -o solve_mumps.so

echo "building petsc"
mpicc -I/usr/include/petsc/ solve_petsc.c -O3 -fPIC -shared -lpetsc -o solve_petsc.so

echo "building amgcl"
git clone https://github.com/ddemidov/amgcl.git
cd amgcl
mkdir build
cd build
cmake -DAMGCL_BUILD_EXAMPLES=ON ..
make libamgcl
cd ../..
g++ solve_amgcl.cpp -O3 -fPIC -shared -o solve_amgcl.so -L./amgcl/build/lib/ -I./amgcl/

echo "building eigen"
EIGEN_FILE=3.3.7.zip
wget http://bitbucket.org/eigen/eigen/get/$EIGEN_FILE
unzip $EIGEN_FILE
EIGEN_DIR=$(ls . | grep eigen-eigen)
g++ solve_eigen.cpp -O3 -fPIC -shared -o solve_eigen.so -I$EIGEN_DIR
