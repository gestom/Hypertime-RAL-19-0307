## Experiments of RAL-19-307

This repo contains code and dataset for the first experiment described in our submission to RAL-19-0307.  This repository allows to re-run the experiments described in the Section 4A. The access to the additional experiments and data will be released after the first round of reviews or on request.

### Prerequisities

The software was tested on <b>Ubuntu 16.04</b> with <b>openCV2.7</b> and <b>Python 2.7.12</b> <b>python-pandas 0.17.1</b>, <b>python-numpy 1.11.0</b> and <b>python-opencv 2.4.9.1</b>.

### Compiling and running

To run it, do the following

1. Install the required packages:
* sudo apt install g++ make libopencv-dev python-opencv python-dev python-numpy python-pandas libalglib-dev transfig gnuplot imagemagick graphviz

2. Compile the test codes:
* cd Hypertime-RAL-19-0307/door_state/src
* make

3. Run the tests
**.cd ../eval_scripts/
**./process_dataset.sh greg_door_2016_min

4. Check the results
* make 
* mkdir tmp
* ./summarize_results.sh greg_door_2016_min
* display summary.png

### Additional, custom models

The temporal models are stored in the *src/models* folder and by implementing the methods of CTemporal.cpp, you can make and test your own method. To re-run the experiments described in Section 4A, with your new model, modify the `test models` file accordingly. Then go to the *src* folder and type **make** to compile the predictive framework. After that, go to the *eval_scripts* folder and type *make* to compile the *t-test* utility.

Then, type:

**./process_dataset.sh greg_door_2016_min**

if it runs well, type

**./summarise_results.sh greg_door_2016_min**

and then check the *summary.png* file, which should look similarly to Fig 2 in the paper.

