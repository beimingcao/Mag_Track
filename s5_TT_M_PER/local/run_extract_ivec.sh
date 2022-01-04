#!/bin/bash
# Copyright 2013   Daniel Povey
#           2014   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

# This example script is still a bit of a mess, and needs to be
# cleaned up, but it shows you all the basic ingredients.

if [ $# -ne 4 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

CV=$1$2
mono_tri=$3

# Config:
gmmdir=$CV/exp/$mono_tri
data_fmllr=$CV/data_delta_ivec
ivec_dim=$4
nGauss=2048
train_nj=10
decode_nj=2

. cmd.sh
. path.sh

for x in train; do 
  dir=$data_fmllr/$x
  steps/nnet/make_delta_feats.sh --nj $train_nj --cmd "$train_cmd" $dir $CV/data/$x $gmmdir $dir/log $dir/data || exit 1;
#  local/ivec/compute_vad_decision.sh --cmd "$train_cmd" --nj 8 $dir $CV/exp/make_vad $dir/data
done
for x in test; do 
  dir=$data_fmllr/$x
  steps/nnet/make_delta_feats.sh --nj $decode_nj --cmd "$train_cmd" $dir $CV/data/$x $gmmdir $dir/log $dir/data || exit 1;
#  local/ivec/compute_vad_decision.sh --cmd "$train_cmd" --nj 8 $dir $CV/exp/make_vad $dir/data
done

# Note: to see the proportion of voiced frames you can do,
# grep Prop $CV/exp/make_vad/vad_*.1.log 


# The recipe currently uses delta-window=3 and delta-order=2. However
# the accuracy is almost as good using delta-window=4 and delta-order=1
# and could be faster due to lower dimensional features.  Alternative
# delta options (e.g., --delta-window 4 --delta-order 1) can be provided to
# sid/train_diag_ubm.sh.  The options will be propagated to the other scripts.
local/ivec/train_diag_ubm.sh --nj $train_nj --cmd "$train_cmd" $data_fmllr/train ${nGauss} \
    $CV/exp/diag_ubm_${mono_tri}_${nGauss}

local/ivec/train_full_ubm.sh --nj $train_nj --cmd "$train_cmd" $data_fmllr/train \
    $CV/exp/diag_ubm_${mono_tri}_${nGauss} $CV/exp/full_ubm_${mono_tri}_${nGauss}

# Train the iVector extractor for training speakers.
local/ivec/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=8G,ram_free=8G" \
  --num-iters 5 --num-processes 1 --nj $train_nj --ivector-dim $ivec_dim $CV/exp/full_ubm_${mono_tri}_${nGauss}/final.ubm $data_fmllr/train $CV/exp/extractor${ivec_dim}_${mono_tri}_${nGauss}

# Extract the iVectors for the training and test data.
local/ivec/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj $train_nj \
   $CV/exp/extractor${ivec_dim}_${mono_tri}_${nGauss} $data_fmllr/train $CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_train

local/ivec/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj $decode_nj \
   $CV/exp/extractor${ivec_dim}_${mono_tri}_${nGauss} $data_fmllr/test $CV/exp/ivectors${ivec_dim}_${mono_tri}_${nGauss}_test


