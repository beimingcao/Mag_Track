#!/bin/bash

for ivec in 20; do
for x in `seq 6`; do
#  ./local/run_blstm_ivec.sh CV$x mono $ivec
  ./local/run_lstm_ivec.sh CV$x tri1 $ivec

#  ./local/run_blstm_basis_fmllr_ivec.sh CV$x mono3 $ivec
  ./local/run_lstm_basis_fmllr_ivec.sh CV$x tri3 $ivec
done
done
