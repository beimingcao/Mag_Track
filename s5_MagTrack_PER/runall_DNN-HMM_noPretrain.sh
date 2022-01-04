#!/bin/bash

for x in `seq 1 6`; do
#  ./local/run_dnn_seq.sh CV$x mono
  ./local/run_dnn_seq_nopretrain.sh CV$x tri1
  for y in diag; do
#    ./local/run_dnn_basis_fmllr.sh CV$x mono3 $y
    ./local/run_dnn_basis_fmllr_nopretrain.sh CV$x tri3 $y
  done
done
