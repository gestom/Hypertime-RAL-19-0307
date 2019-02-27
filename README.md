# Hypertime

This repo contains code and dataset for the first experiment described in our submission to RAL-19-0307. The access to the additional experiments and data will be released after the first round of reviews or on request. The software was tested on <b>Ubuntu 16.04</b> with <b>openCV2.7</b> and <b>Python</b>.

To run it, 


sudo apt install g++ make libopencv-dev python-opencv python-dev python-numpy python-pandas libalglib-dev transfig gnuplot imagemagick graphviz

cd Hypertime-RAL-19-0307/door_state/src

make

cd models/

vim test_models.txt

// modify it accordingly:

FreMEn 1 2 3 4 5

HyT-KM 0

HyT-EM 1 2 3 4 5

Mean 0

#None 0

Hist 1 4 12 24

#GMM 1 2 3 4 5

#VonMises 1 2 3 4 5

//

cd ../../eval_scripts/

./process_dataset.sh greg_door_2016_min

./summarize_results.sh greg_door_2016_min

display summary.png
