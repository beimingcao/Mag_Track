#!/bin/bash 

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
mfcc_config=conf/mfcc.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <path-to-mvdir>";
   echo "e.g.: $0 data/train exp/make_mfcc/train mfcc"
   echo "options: "
   echo "  --mfcc-config <config-file>                      # config passed to compute-mfcc-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
mvdir=$3


# make $mvdir an absolute pathname.
mvdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mvdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mvdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/mv.scp

required="$scp $mfcc_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $mvdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mvdir/raw_mv_$name.$n.ark
done


if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done
 
  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_mv_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    compute-mfcc-feats $vtln_opts --verbose=2 --config=$mfcc_config ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$mvdir/raw_mfcc_$name.JOB.ark,$mvdir/raw_mfcc_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming mv.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/mv_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;


  # add ,p to the input rspecifier so that we can just skip over
  # utterances that have bad wave data.
  #feats_one="$(cat $logdir/mv_${name}.JOB.scp | awk '{print $2}')"
	#mv_feats="ark:copy-feats ark:mv/raw_mv_$name.JOB.ark ark:- |"
  #$cmd JOB=1:$nj $logdir/make_mfcc_${name}.JOB.log \
  #  cat `cat $logdir/mv_${name}.JOB.scp | awk '{print $2}'` | \
  #    copy-feats --compress=$compress ark:- \
  #    ark,scp:$mvdir/raw_mv_$name.JOB.ark,$mvdir/raw_mv_$name.JOB.scp \
  #    || exit 1;
	for x in `seq $nj`; do
    cat $(cat $logdir/mv_${name}.$x.scp | awk '{print $2}') | \
      copy-feats --compress=$compress ark:- \
      ark,scp:$mvdir/raw_mv_$name.$x.ark,$mvdir/raw_mv_$name.$x.scp \
      || exit 1;
	done 
fi

    #compute-mfcc-feats  $vtln_opts --verbose=2 --config=$mfcc_config \
    # scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \

if [ -f $logdir/.error.$name ]; then
  echo "Error producing mfcc features for $name:"
  tail $logdir/make_mv_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $mvdir/raw_mv_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/mv_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating MFCC features for $name"
