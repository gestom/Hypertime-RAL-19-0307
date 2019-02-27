# Hypertime

This repo contains code and dataset for the first experiment described in our submission to RAL-19-0307. The access to the additional experiments and data will be released after the first round of reviews or on request. The software was tested on <b>Ubuntu 16.04</b> with <b>openCV2.7</b> and <b>Python 2.7.12</b> <b>python-pandas 0.17.1</b>, <b>python-numpy 1.11.0</b> and <b>
python-opencv 2.4.9.1</b>.

To run it, do the following

1. Install the required packages:
* sudo apt install g++ make libopencv-dev python-opencv python-dev python-numpy python-pandas libalglib-dev transfig gnuplot imagemagick graphviz

2. Compile the test codes:
* cd Hypertime-RAL-19-0307/door_state/src
* make

3. Run the tests
* cd ../eval_scripts/
* ./process_dataset.sh greg_door_2016_min

4. Check the results
* make 
* mkdir tmp
* ./summarize_results.sh greg_door_2016_min
* display summary.png

// to run only the desired models and parameters, modify the `test models` file accordingly
