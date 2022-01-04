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

if [ $# -ne 4 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi
CV=$1
mono_tri=$2
full_diag=$3

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=$CV/exp/$mono_tri
data_fmllr=$CV/data_fmllr
ivec_dim=$4
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

	#dir=$data_fmllr/test
  #steps/nnet/make_delta_feats.sh --nj 4 --cmd "$train_cmd" \
  #   $dir $CV/data/test $gmmdir $dir/log $dir/data || exit 1 

	dir=$data_fmllr/test
  steps/nnet/make_basis_fmllr_feats.sh --nj $decode_nj --cmd "$train_cmd" \
    --transform-dir $CV/exp/$mono_tri/decode_test \
    $dir $CV/data/test $gmmdir $dir/log $dir/data || exit 1 
  # dev
#  dir=$data_fmllr/dev
#  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
#     --transform-dir $gmmdir/decode_dev \
#     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train

	#steps/nnet/make_delta_feats.sh --nj 4 --cmd "$train_cmd" \
  #   $dir $CV/data/train $gmmdir $dir/log $dir/data || exit 1 

	steps/nnet/make_basis_fmllr_feats.sh --nj $train_nj --cmd "$train_cmd" \
    --transform-dir $CV/exp/$mono_tri \
     $dir $CV/data/train $gmmdir $dir/log $dir/data || exit 1 

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

# Create ark with dummy-ivectors,
#[ ! -e data/dummy_ivec.ark ] && cat {$train,$dev}/feats.scp | awk '{ print $1, "[ 0 0 0 ]"; }' >data/dummy_ivec.ark
ivector_train=scp:$CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_train/ivector.scp
ivector_test=scp:$CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_test/ivector.scp

# Build NN, with pre-training (script test),
if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=$CV/exp/dnn4h-ivec${ivec_dim}_${nGauss}_pretrain-dbn-$mono_tri
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh \
      --ivector $ivector_train --splice 4 --hid-dim 512 --rbm-iter 10 $data_fmllr/train $dir
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
	for x in `seq 1 6`; do
  dir=$CV/exp/dnn4h-ivec${ivec_dim}_${nGauss}_pretrain-dbn-${mono_tri}_dnn$x
  ali=$CV/exp/dnn4_pretrain-dbn-${mono_tri}_dnn5_ali
  #ali=$CV/exp/dnn4_pretrain-dbn-${mono_tri}_dnn5_ali
  feature_transform=$CV/exp/dnn4h-ivec${ivec_dim}_${nGauss}_pretrain-dbn-$mono_tri/final.feature_transform
  dbn=$CV/exp/dnn4h-ivec${ivec_dim}_${nGauss}_pretrain-dbn-$mono_tri/${x}.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    --ivector $ivector_train \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 $CV/data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    --ivector $ivector_test \
    $gmmdir/graph $data_fmllr/test $dir/decode_test_dbn$x
	done
fi
exit 0;

# Sequence training using sMBR criterion, we do Stochastic-GD with per-utterance updates.
# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2, 
# the value 0.1 is better both for decoding and sMBR.
dir=exp/dnn4h-dummy-ivec_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4h-dummy-ivec_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 4 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 5 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 3 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --ivector $ivector \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1
  done 
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
