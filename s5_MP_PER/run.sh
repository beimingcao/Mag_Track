#!/bin/bash

#
# Copyright 2013 Bagher BabaAli,
#           2014 Brno University of Technology (Author: Karel Vesely)
#
# TIMIT, description of the database:
# http://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
#
# Hon and Lee paper on TIMIT, 1988, introduces mapping to 48 training phonemes, 
# then re-mapping to 39 phonemes for scoring:
# http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
#

if [ $# -ne 2 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

CV=$1$2

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=1000
numGaussTri1=7000
numLeavesMLLT=1000
numGaussMLLT=7000
numLeavesSAT=1000
numGaussSAT=7000
numGaussUBM=250
numLeavesSGMM=1000
numGaussSGMM=2500



#feats_nj=10
#train_nj=30
#decode_nj=5
feats_nj=4
train_nj=7
decode_nj=1

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/home/beiming/kaldi/egs/Mag_Track/CV_all
CV=$1$2

local/timit_data_prep_CV.sh $timit $1 $2 || exit 1

local/timit_prepare_dict.sh $CV

# Caution below: we insert optional-silence with probability 0.5, which is the
# default, but this is probably not appropriate for this setup, since silence
# appears also as a word in the dictionary and is scored.  We could stop this
# by using the option --sil-prob 0.0, but apparently this makes results worse.
# (-> In sclite scoring the deletions of 'sil' are not scored as errors)
#utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
# $CV/data/local/dict "sil" $CV/data/local/lang_tmp $CV/data/lang

utils/prepare_lang.sh $CV/data/local/dict "<UNK>" $CV/data/local/lang_tmp $CV/data/lang

local/timit_format_data.sh $CV
echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=$CV/mfcc

for x in train; do 
  steps/make_MotionVector.sh --cmd "$train_cmd" --nj $train_nj $CV/data/$x $CV/exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh $CV/data/$x $CV/exp/make_mfcc/$x $mfccdir
done

for x in test; do 
  steps/make_MotionVector.sh --cmd "$train_cmd" --nj $decode_nj $CV/data/$x $CV/exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh $CV/data/$x $CV/exp/make_mfcc/$x $mfccdir
done

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" --cmvn-opts "--norm-vars=false" $CV/data/train $CV/data/lang $CV/exp/mono

utils/mkgraph.sh --mono $CV/data/lang_test_bg $CV/exp/mono $CV/exp/mono/graph

#steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/mono/graph data/dev exp/mono/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 $CV/exp/mono/graph $CV/data/test $CV/exp/mono/decode_test

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 $CV/data/train $CV/data/lang $CV/exp/mono $CV/exp/mono_ali

steps/train_sat_basis.sh --cmd "$train_cmd" --train-tree false \
 $numLeavesSAT $numLeavesSGMM $CV/data/train $CV/data/lang $CV/exp/mono_ali $CV/exp/mono3
 #$numLeavesSAT $numGaussSAT $CV/data/train $CV/data/lang $CV/exp/mono_ali $CV/exp/mono3

utils/mkgraph.sh --mono $CV/data/lang_test_bg $CV/exp/mono3 $CV/exp/mono3/graph

#steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri3/graph data/dev exp/tri3/decode_dev

: <<'END'
for x in diag; do
  steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --fmllr_update_type $x --fmllr_min_count 50 \
    $CV/exp/mono3/graph $CV/data/test $CV/exp/mono3/decode_test_$x 
done 
END
  steps/decode_basis_fmllr_utt.sh --nj "$decode_nj" --cmd "$decode_cmd" \
    $CV/exp/mono3/graph $CV/data/test $CV/exp/mono3/decode_test

steps/align_basis_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $CV/data/train $CV/data/lang $CV/exp/mono3 $CV/exp/mono3_ali
 
# Train tri1, which is deltas + delta-deltas, on train data.
steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-vars=false" \
 $numLeavesTri1 $numGaussTri1 $CV/data/train $CV/data/lang $CV/exp/mono_ali $CV/exp/tri1

utils/mkgraph.sh $CV/data/lang_test_bg $CV/exp/tri1 $CV/exp/tri1/graph

#steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri1/graph data/dev exp/tri1/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 $CV/exp/tri1/graph $CV/data/test $CV/exp/tri1/decode_test

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
  $CV/data/train $CV/data/lang $CV/exp/tri1 $CV/exp/tri1_ali

: <<'END'
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 --splice-opts "--left-context=3 --right-context=3" \
 $numLeavesMLLT $numGaussMLLT $CV/data/train $CV/data/lang $CV/exp/tri1_ali $CV/exp/tri2

utils/mkgraph.sh $CV/data/lang_test_bg $CV/exp/tri2 $CV/exp/tri2/graph

#steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri2/graph data/dev exp/tri2/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 $CV/exp/tri2/graph $CV/data/test $CV/exp/tri2/decode_test 
END

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

: <<'END'
# Align tri2 system with train data.
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true $CV/data/train $CV/data/lang $CV/exp/tri2 $CV/exp/tri2_ali
END

# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat_basis.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT $CV/data/train $CV/data/lang $CV/exp/tri1_ali $CV/exp/tri3

utils/mkgraph.sh $CV/data/lang_test_bg $CV/exp/tri3 $CV/exp/tri3/graph

#steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri3/graph data/dev exp/tri3/decode_dev

: <<'END'
for x in diag; do
  steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --fmllr_update_type $x --fmllr_min_count 50 \
    $CV/exp/tri3/graph $CV/data/test $CV/exp/tri3/decode_test_$x 
done 
END
  steps/decode_basis_fmllr_utt.sh --nj "$decode_nj" --cmd "$decode_cmd" \
    $CV/exp/tri3/graph $CV/data/test $CV/exp/tri3/decode_test

echo ============================================================================
echo "                        SGMM2 Training & Decoding                         "
echo ============================================================================

steps/align_basis_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $CV/data/train $CV/data/lang $CV/exp/tri3 $CV/exp/tri3_ali
 
exit 0;
: <<'END'
#exit 0 # From this point you can run DNN : local/run_dnn.sh

steps/train_ubm.sh --cmd "$train_cmd" \
 $numGaussUBM $CV/data/train $CV/data/lang $CV/exp/tri3_ali $CV/exp/ubm4

steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
 $CV/data/train $CV/data/lang $CV/exp/tri3_ali $CV/exp/ubm4/final.ubm $CV/exp/sgmm2_4

utils/mkgraph.sh $CV/data/lang_test_bg $CV/exp/sgmm2_4 $CV/exp/sgmm2_4/graph

#steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
# --transform-dir exp/tri3/decode_dev exp/sgmm2_4/graph data/dev \
# exp/sgmm2_4/decode_dev

steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
 --transform-dir $CV/exp/tri3/decode_test $CV/exp/sgmm2_4/graph $CV/data/test \
 $CV/exp/sgmm2_4/decode_test 
END

echo ============================================================================
echo "                    MMI + SGMM2 Training & Decoding                       "
echo ============================================================================

: <<'END'
steps/align_sgmm2.sh --nj "$train_nj" --cmd "$train_cmd" \
 --transform-dir exp/tri3_ali --use-graphs true --use-gselect true \
 data/train data/lang exp/sgmm2_4 exp/sgmm2_4_ali

steps/make_denlats_sgmm2.sh --nj "$train_nj" --sub-split "$train_nj" \
 --acwt 0.2 --lattice-beam 10.0 --beam 18.0 \
 --cmd "$decode_cmd" --transform-dir exp/tri3_ali \
 data/train data/lang exp/sgmm2_4_ali exp/sgmm2_4_denlats

steps/train_mmi_sgmm2.sh --acwt 0.2 --cmd "$decode_cmd" \
 --transform-dir exp/tri3_ali --boost 0.1 --drop-frames true \
 data/train data/lang exp/sgmm2_4_ali exp/sgmm2_4_denlats exp/sgmm2_4_mmi_b0.1

for iter in 1 2 3 4; do
#  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
#   --transform-dir exp/tri3/decode_dev data/lang_test_bg data/dev \
#   exp/sgmm2_4/decode_dev exp/sgmm2_4_mmi_b0.1/decode_dev_it$iter

  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_test data/lang_test_bg data/test \
   exp/sgmm2_4/decode_test exp/sgmm2_4_mmi_b0.1/decode_test_it$iter
done 
END

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

: <<'END'
# DNN hybrid system training parameters
dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 \
  --final-learning-rate 0.002 --num-hidden-layers 2  \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  data/train data/lang exp/tri3_ali exp/tri4_nnet

#[ ! -d exp/tri4_nnet/decode_dev ] && mkdir -p exp/tri4_nnet/decode_dev
decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
#steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
#  --transform-dir exp/tri3/decode_dev exp/tri3/graph data/dev \
#  exp/tri4_nnet/decode_dev | tee exp/tri4_nnet/decode_dev/decode.log

[ ! -d exp/tri4_nnet/decode_test ] && mkdir -p exp/tri4_nnet/decode_test
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  --transform-dir exp/tri3/decode_test exp/tri3/graph data/test \
  exp/tri4_nnet/decode_test | tee exp/tri4_nnet/decode_test/decode.log 
END

echo ============================================================================
echo "                    System Combination (DNN+SGMM)                         "
echo ============================================================================

: <<'END'
for iter in 1 2 3 4; do
#  local/score_combine.sh --cmd "$decode_cmd" \
#   data/dev data/lang_test_bg exp/tri4_nnet/decode_dev \
#   exp/sgmm2_4_mmi_b0.1/decode_dev_it$iter exp/combine_2/decode_dev_it$iter

  local/score_combine.sh --cmd "$decode_cmd" \
   data/test data/lang_test_bg exp/tri4_nnet/decode_test \
   exp/sgmm2_4_mmi_b0.1/decode_test_it$iter exp/combine_2/decode_test_it$iter
done 
END

echo ============================================================================
echo "               DNN Hybrid Training & Decoding (Karel's recipe)            "
echo ============================================================================

#for iter in 1 2 3 4; do
#  local/score_combine.sh --cmd "$decode_cmd" \
#   data/test data/lang_test_bg exp/tri4_nnet/decode_test \
#   exp/sgmm2_4_mmi_b0.1/decode_test_it$iter exp/combine_2/decode_test_it$iter
#done 
#exit 0;
local/run_dnn.sh $CV mono
local/run_dnn.sh $CV tri1
exit 0;

echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

#bash RESULTS dev
bash RESULTS test

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
