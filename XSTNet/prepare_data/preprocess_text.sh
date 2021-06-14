#!/usr/bin/env bash

set -e

function makeDirs(){
    if [[ $1 == hdfs://* ]]; then
        hadoop fs -mkdir -p $1
    else
        mkdir -p $1
    fi
}

DATA_PATH=$1
VOCAB_PATH=$2
TGT_LANG=$3
shift 3



echo " === Tokenize Text Using SPM and make TFRecord  ==="

TRANSCRIPT_PATH=${DATA_PATH}/transcripts
OUT_MT_TFRECORD_PATH=${DATA_PATH}/asr_st
makeDirs ${OUT_MT_TFRECORD_PATH}/train


sed -e "s#TRANSCRIPT_PATH#${TRANSCRIPT_PATH}#" -e "s#VOCAB_PATH#${VOCAB_PATH}#" -e "s#TGT_LANG#${TGT_LANG}#"  -e "s#SUBSET#train#" config/text_data_preprocessing.yml > config_processed.yml

PROCESSORS_IN_PARALLEL=4
NUM_PROCESSORS=4
TOTAL_SHARDS=64
SHARD_PER_PROCESS=$((TOTAL_SHARDS / NUM_PROCESSORS)) # 8
LOOP=$((NUM_PROCESSORS / PROCESSORS_IN_PARALLEL))

for loopid in $(seq 1 ${LOOP}); do
    start=$(($((loopid - 1)) * ${PROCESSORS_IN_PARALLEL}))
    end=$(($start + PROCESSORS_IN_PARALLEL - 1))
    echo ${start}, ${end}
    for procid in $(seq ${start} ${end}); do
        set -x
        nice -n 10 python3 -m bytedseq.cli.create_tfrecords \
            --processor_id ${procid} --num_processors ${NUM_PROCESSORS} \
            --num_output_shards ${TOTAL_SHARDS} \
            --output_range_begin "$((SHARD_PER_PROCESS * procid))" \
            --output_range_end "$((SHARD_PER_PROCESS * procid + SHARD_PER_PROCESS))" \
        --config_paths config_processed.yml \
        --output_template ${OUT_MT_TFRECORD_PATH}/train/train.tfrecords-%5.5d-of-%5.5d || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done

makeDirs ${OUT_MT_TFRECORD_PATH}/devtest
sed -e "s#TRANSCRIPT_PATH#${TRANSCRIPT_PATH}#" -e "s#VOCAB_PATH#${VOCAB_PATH}#" -e "s#TGT_LANG#${TGT_LANG}#"  -e "s#SUBSET#dev#" config/text_data_preprocessing.yml > config_processed.yml
nice -n 10 python3 -m bytedseq.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
    --config_paths config_processed.yml \
    --output_template ${OUT_MT_TFRECORD_PATH}/devtest/dev.en-${TGT_LANG}.tfrecords-%5.5d-of-%5.5d

sed -e "s#TRANSCRIPT_PATH#${TRANSCRIPT_PATH}#" -e "s#VOCAB_PATH#${VOCAB_PATH}#" -e "s#TGT_LANG#${TGT_LANG}#"  -e "s#SUBSET#tst-COMMON#" config/text_data_preprocessing.yml > config_processed.yml
nice -n 10 python3 -m bytedseq.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
    --config_paths config_processed.yml \
    --output_template ${OUT_MT_TFRECORD_PATH}/devtest/tst-COMMON.en-${TGT_LANG}.tfrecords-%5.5d-of-%5.5d
