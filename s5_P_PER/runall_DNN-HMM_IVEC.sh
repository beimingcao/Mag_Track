#!/bin/bash

for ivec in 20; do
for x in `seq 6`; do
#  ./local/run_extract_ivec.sh CV $x tri1 $ivec
#  ./local/run_extract_basis_fmllr_ivec.sh CV $x tri3 $ivec

#  ./local/run_dnn_ivec.sh CV$x mono $ivec
#  ./local/run_dnn_ivec.sh CV$x tri1 $ivec
  for y in diag; do
#    ./local/run_dnn_basis_fmllr_ivec.sh CV$x mono3 $y $ivec
    ./local/run_dnn_basis_fmllr_ivec.sh CV$x tri3 $y $ivec
  done
done
done
