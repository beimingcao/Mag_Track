#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a BLSTM network on FBANK features.
# The initial BLSTM code comes from Ni Chongjia (I2R), thanks!

# We use multi-stream training, while the BPTT is done over whole
# utterances with similar length (selection done with C++ class MatrixBuffer).

# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2,
# the value 0.1 is better both for decoding and sMBR.

. ./cmd.sh
. ./path.sh

if [ $# -ne 2 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi
CV=$1
mono_tri=$2

# Config:
gmmdir=$CV/exp/$mono_tri
data_fmllr=$CV/$mono_tri/data_fmllr
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
  steps/nnet/make_fmllr_feats.sh --nj $decode_nj --cmd "$train_cmd" \
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

	steps/nnet/make_fmllr_feats.sh --nj $train_nj --cmd "$train_cmd" \
    --transform-dir $CV/exp/$mono_tri \
     $dir $CV/data/train $gmmdir $dir/log $dir/data || exit 1 

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=$CV/exp/blstm4-${mono_tri}_splice0_nstream10_celdim320_pdim200_nlayer2
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
      --proto-opts "--cell-dim 320 --proj-dim 200 --num-layers 2" \
    ${train}_tr90 ${train}_cv10 $CV/data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $test $dir/decode_test 
fi

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
