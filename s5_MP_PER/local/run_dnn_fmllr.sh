#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

if [ $# -ne 3 ]; then
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
#data_fmllr=$CV/data_delta
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
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=$CV/exp/dnn4_pretrain-dbn-$mono_tri
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 6 --hid-dim 512 --splice 4 --rbm-iter 10 $data_fmllr/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
	for x in `seq 1 6`; do
		dir=$CV/exp/dnn4_pretrain-dbn-${mono_tri}_dnn$x
  
    if [ $x == 1 ]; then
  		ali=${gmmdir}_ali
    else
      y=$((x-1))
      srcdir=$CV/exp/dnn4_pretrain-dbn-${mono_tri}_dnn$y
      steps/nnet/align.sh --nj $train_nj --cmd "$train_cmd" \
      $data_fmllr/train $CV/data/lang $srcdir ${srcdir}_ali || exit 1;
      ali=${srcdir}_ali
    fi
		
		feature_transform=$CV/exp/dnn4_pretrain-dbn-$mono_tri/final.feature_transform
		dbn=$CV/exp/dnn4_pretrain-dbn-$mono_tri/${x}.dbn
		(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
		# Train
		$cuda_cmd $dir/log/train_nnet.log \
			steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
			$data_fmllr/train_tr90 $data_fmllr/train_cv10 $CV/data/lang $ali $ali $dir || exit 1;
		# Decode (reuse HCLG graph)
		steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
		  $gmmdir/graph $data_fmllr/test $dir/decode_test_dbn${x} #|| exit 1;
	done
#  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
#    $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
fi
exit 0;
