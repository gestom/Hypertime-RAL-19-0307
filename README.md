# Hypertime

This repo contains code and dataset for the first experiment described in our submission to RAL-18-0259. The access to the additional experiments and data will be released after the first round of reviews or on request.



How to run that:
for Ubuntu 16.04 (tested on fresh Lubuntu and Xubuntu 16.04.3):

sudo apt install g++ make

sudo apt install libopencv-dev python-opencv

sudo apt install python-dev python-numpy python-pandas

sudo apt install libalglib-dev transfig gnuplot imagemagick graphviz

cd Hypertime-RAL-18-0278/door_state/src

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

