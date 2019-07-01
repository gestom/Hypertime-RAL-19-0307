set terminal fig color 
set xlabel 'Dataset [-]' offset 0.0,0.2
set ylabel 'Mean squared error [-]' offset 1.2,0.0
set size 0.65,0.75
set title 'Prediction error rate of individiual models'
set key top horizontal 
#set xtics ("Training" 0, "Testing 1" 1, "Testing 2" 2);
set xtics ("Training" 0);
set style fill transparent solid 1.0 noborder;
set boxwidth XXX 
plot [-0.5:0.5] [0:0.2]\
