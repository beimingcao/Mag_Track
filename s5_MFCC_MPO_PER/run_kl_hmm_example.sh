#!/bin/bash

# Copyright 2013 Idiap Research Institute (Author: David Imseng)
# Apache 2.0

. cmd.sh
. path.sh

states=10000


for n in 1 2 3; do
	for x in 1 2 3 4 5 6; do

## monophone DNN system
dir_mono=CV$n/exp/dnn4_pretrain-dbn-mono_dnn$x
gmm_mono=CV$n/exp/mono

## triphone DNN system
dir=CV$n/exp/dnn4_pretrain-dbn-tri1_dnn$x
dir_ali=CV$n/exp/dnn4_pretrain-dbn-tri1_dnn$x
gmm=CV$n/exp/tri1

data_train=CV$n/data_delta/train
data_test=CV$n/data_delta/test

# clustered state (clustered triphone) generation in original script, but we do not use this function 
#####steps/kl_hmm/build_tree.sh --cmd "$big_memory_cmd" --thresh -1 --nnet_dir ${dir} \
##### ${states} data/train_si284 data/lang exp/tri2a_ali_si284 exp/tri5b-${states} || exit 1;

### acoustic unit: monophone, lexical unit: monophone (acoustic unit means the output of DNN, lexical unit means the HMM states 
# state alignment
steps/nnet/align.sh --nj 1 $data_train CV$n/data/lang $dir_mono ${dir_mono}_ali 

# KL-HMM training 
steps/kl_hmm/train_kl_hmm.sh --nj 1 --cmd "$big_memory_cmd" --nnet $dir_mono/final.nnet --model $dir_mono/final.mdl $data_train ${dir_mono}_ali $dir_mono/kl-hmm-mono

# Decoding 
steps/kl_hmm/decode_kl_hmm.sh --nj 1 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir_mono/kl-hmm-mono/final.nnet --model $dir_mono/final.mdl \
	--config conf/decode_dnn.config $gmm_mono/graph/ $data_test $dir_mono/decode_kl-hmm-mono 
### End --------------------------------------------

### acoustic unit: triphone, lexical unit: monophone  
# KL-HMM training 
#steps/kl_hmm/train_kl_hmm.sh --nj 8 --cmd "$big_memory_cmd" --nnet $dir/final.nnet --model $dir_mono/final.mdl $data_train ${dir_mono}_ali $dir/kl-hmm-mono

# Decoding 
#steps/kl_hmm/decode_kl_hmm.sh --nj 8 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir/kl-hmm-mono/final.nnet --model $dir_mono/final.mdl \
#	--config conf/decode_dnn.config $gmm_mono/graph/ $data_test $dir/decode_kl-hmm-mono
### End --------------------------------------------------

### acoustic unit: monophone, lexical unit: triphone
# state alignment
#steps/nnet/align.sh --nj 8 data-fbank/train data/lang $dir ${dir}_ali

# KL-HMM training 
#steps/kl_hmm/train_kl_hmm.sh --nj 8 --cmd "$big_memory_cmd" --nnet $dir_mono/final.nnet --model $dir/final.mdl $data_train ${dir}_ali $dir_mono/kl-hmm-tri

# Decoding
#steps/kl_hmm_decode_kl_hmm.sh --nj 8 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir_mono/kl-hmm-tri/final.net --model $dir/final.mdl \
#	--config conf/decode_dnn.config $gmm/graph/ $data_test $dir_mono/decode_kl-hmm-tri
### End ----------------------------------------------

### acoustic unit: triphone, lexical unit: triphone
# KL-HMM training
steps/kl_hmm/train_kl_hmm.sh --nj 1 --cmd "$big_memory_cmd" --nnet $dir/final.nnet --model $dir/final.mdl $data_train ${dir}_ali $dir/kl-hmm-tri

# Decoding
steps/kl_hmm/decode_kl_hmm.sh --nj 1 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir/kl-hmm-tri/final.nnet --model $dir/final.mdl \
	--config conf/decode_dnn.config $gmm/graph/ $data_test $dir/decode_kl-hmm-tri
### End --------------------------------------------

	done
done
