# Benchmarks

Running the benchmarks will take about a day, not including the time required to install all libraries.

```bash
# install required libraries for benchmark
sudo apt install build-essential unzip cmake libboost-all-dev libopenmpi-dev libmumps-dev petsc-dev libsuitesparse-dev swig
pip3 install psutil scikit-umfpack pyamg pymatting natsort

# download pymatting
git clone https://github.com/pymatting/pymatting
cd pymatting
# download test images
python3 tests/download_images.py

# build libraries
cd benchmarks
sh rebuild.sh

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

python3 calculate_laplacian_error.py
python3 plot_laplacian_error_per_image.py

python3 benchmark_image_sizes.py
python3 plot_results.py
```

If you should be unable to install a specific solver, you can disable it by removing the corresponding line from the list `SOLVER_NAMES` in `pymatting/benchmarks/config.py`.

For faster debugging, it might be helpful to uncomment `SCALES` in the config file.

Sometimes it helps to build packages from source instead of binaries:

```bash
pip3 uninstall <package>
pip3 install <package> --no-binary :all:
```
