### Section 5A of RAL-18-0278

To re-run the experiments described in Section 5A, first go to the *src* folder and type **make** to compile the predictive framework.
Then, go to the *eval_scripts* folder and type *make* to compile the *t-test* utility.

Then, type:

**./process_dataset.sh greg_door_2016_min**

if it runs well, type

**./summarise_results.sh greg_door_2016_min**

and then check the *summary.png* file, which should look similarly to Fig 2 in the paper.

You will need to install openCV2.7, libalglib, gnuplot and transfig to run the software.
The temporal models are stored in the *src/models* folder and by implementing the methods of CTemporal.cpp, you can make and test your own method.
