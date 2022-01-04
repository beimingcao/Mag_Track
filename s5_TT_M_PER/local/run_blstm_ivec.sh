#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example demonstrates how to add i-vector on DNN input (or any other side-info). 
# A fixed vector is pasted to all the frames of an utterance and forwarded to nn-input `as-is', 
# bypassing both the feaure transform and global CMVN normalization.
#
# The i-vector is simulated by a dummy vector [ 0 0 0 ],
# note that all the scripts get an extra option '--ivector'
#
# First we train NN with w/o RBM pre-training, then we do the full recipe:
# RBM pre-training, per-frame training, and sequence-discriminative training.

# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2, 
# the value 0.1 is better both for decoding and sMBR.

if [ $# -ne 3 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi
CV=$1
mono_tri=$2

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=$CV/exp/$mono_tri
data_fmllr=$CV/data_delta
ivec_dim=$3
nGauss=2048
stage=0 # resume training with --stage=N
train_nj=10
decode_nj=2
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test

	dir=$data_fmllr/test
  steps/nnet/make_delta_feats.sh --nj $decode_nj --cmd "$train_cmd" \
     $dir $CV/data/test $gmmdir $dir/log $dir/data || exit 1 

  # dev
#  dir=$data_fmllr/dev
#  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
#     --transform-dir $gmmdir/decode_dev \
#     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train

	steps/nnet/make_delta_feats.sh --nj $train_nj --cmd "$train_cmd" \
     $dir $CV/data/train $gmmdir $dir/log $dir/data || exit 1 

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

# Create ark with dummy-ivectors,
#[ ! -e data/dummy_ivec.ark ] && cat {$train,$dev}/feats.scp | awk '{ print $1, "[ 0 0 0 ]"; }' >data/dummy_ivec.ark
ivector_train=scp:$CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_train/ivector.scp
ivector_test=scp:$CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_test/ivector.scp

for cell in 320; do
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=$CV/exp/blstm4-ivec${ivec_dim}-${mono_tri}_${nGauss}_splice0_nstream10_celdim${cell}_pdim200_nlayer2_lrate0.00004
  ali=$CV/exp/dnn4_pretrain-dbn-${mono_tri}_dnn5_ali
  train=$data_fmllr/train
  test=$data_fmllr/test

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --network-type blstm --learn-rate 0.00004 \
      --feat-type plain --splice 0 \
      --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-multistream-perutt" \
      --train-tool-opts "--num-streams=10 --max-frames=15000" \
      --proto-opts "--cell-dim ${cell} --proj-dim 200 --num-layers 2" \
      --ivector $ivector_train \
    ${train}_tr90 ${train}_cv10 $CV/data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    --ivector $ivector_test \
    $gmmdir/graph $test $dir/decode_test 
fi
done
# TODO : sequence training,

echo Success
exit 0


