***************
Getting Started
***************

Requirements
############

* numpy>=1.16.0
* pillow>=5.2.0
* numba>=0.44.0
* scipy>=1.1.0

Additional Requirements (for GPU support)
#########################################

* cupy-cuda90>=6.5.0 or similar
* pyopencl>=2019.1.2

Installation
############
To install PyMatting simply run:

.. code-block::
      
   git clone https://github.com/pymatting/pymatting
   cd pymatting
   pip3 install .

Testing
#######
Run the tests from the main directory:

.. code-block::

   python3 tests/download_images.py
   pip3 install -r requirements_tests.txt
   pytest
   
To skip the tests of the GPU implementation use the --no_gpu option.
