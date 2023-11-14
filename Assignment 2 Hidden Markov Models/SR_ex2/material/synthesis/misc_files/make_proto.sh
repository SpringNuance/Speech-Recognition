#!/bin/bash

if (( $# == 0 )) ; then 
    echo ""
    echo "Usage: $0 stream_length"
    echo ""
    exit 1
fi


order=$1

echo "~o
<STREAMINFO> 4 $order 1 1 1
<MSDINFO> 4 0 1 1 1
<VECSIZE> 51 <NULLD><USER><DIAGC>
~h \"proto\"
<BEGINHMM>
<NUMSTATES> 7"


for state in {2..6}; do 
echo "<STATE> $state
<STREAM> 1
<MEAN> $order"
for (( n=1; n<=order; n++ )); do 
 printf "%1.1f " 0;
done
echo ""
echo "<Variance> $order"
for (( n=1; n<=order; n++ )); do 
 printf "%1.1f " 1;
done
echo ""
echo "<STREAM> 2
<NUMMIXES> 2
<MIXTURE> 1 0.5000
<MEAN> 1
1.0
<VARIANCE> 1
1.0
<MIXTURE> 2 0.5000
<MEAN> 0
<VARIANCE> 0
<STREAM> 3
<NUMMIXES> 2
<MIXTURE> 1 0.5000
<MEAN> 1
1.0
<VARIANCE> 1
1.0
<MIXTURE> 2 0.5000
<MEAN> 0
<VARIANCE> 0
<STREAM> 4
<NUMMIXES> 2
<MIXTURE> 1 0.5000
<MEAN> 1
1.0
<VARIANCE> 1
1.0
<MIXTURE> 2 0.5000
<MEAN> 0
<VARIANCE> 0"

done

echo "  <TransP> 7
   0.000e+0   1.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0
   0.000e+0   6.000e-1   4.000e-1   0.000e+0   0.000e+0   0.000e+0   0.000e+0
   0.000e+0   0.000e+0   6.000e-1   4.000e-1   0.000e+0   0.000e+0   0.000e+0
   0.000e+0   0.000e+0   0.000e+0   6.000e-1   4.000e-1   0.000e+0   0.000e+0
   0.000e+0   0.000e+0   0.000e+0   0.000e+0   6.000e-1   4.000e-1   0.000e+0
   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0   6.000e-1   4.000e-1
   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0
<ENDHMM>"
