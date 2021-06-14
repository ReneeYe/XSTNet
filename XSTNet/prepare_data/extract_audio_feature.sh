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
TGT_LANG=$2
shift 2

RAW_DATA_PATH=$DATA_PATH/raw/
INPUT_TARBALL=${RAW_DATA_PATH}/MUSTC_v1.0_en-${TGT_LANG}.tar.gz
OUT_PATH=${DATA_PATH}/en-${TGT_LANG}
OUT_AUDIO_TFRECORD_PATH=${OUT_PATH}/audio
OUT_TRANSCRIPT_PATH=${OUT_PATH}/transcripts

makeDirs ${OUT_TRANSCRIPT_PATH}


echo "=== First pass, collecting transcripts ==="

set -x
python3 -m bytedseq.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction train --add_language_tag \
    --input_tarball ${INPUT_TARBALL} \
    --output_transcript_file ${OUT_TRANSCRIPT_PATH}/train.en.txt \
    --output_translation_file  ${OUT_TRANSCRIPT_PATH}/train.${TGT_LANG}.txt &

python3 -m bytedseq.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction dev --add_language_tag \
    --input_tarball ${INPUT_TARBALL} \
    --output_transcript_file ${OUT_TRANSCRIPT_PATH}/dev.en.txt \
    --output_translation_file  ${OUT_TRANSCRIPT_PATH}/dev.${TGT_LANG}.txt &

python3 -m bytedseq.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction tst-COMMON --add_language_tag\
    --input_tarball ${INPUT_TARBALL} \
    --output_transcript_file ${OUT_TRANSCRIPT_PATH}/tst-COMMON.en.txt \
    --output_translation_file  ${OUT_TRANSCRIPT_PATH}/tst-COMMON.${TGT_LANG}.txt &

wait
set +x

echo "=== Second pass, generating TF Records with audio features and raw transcripts ==="
makeDirs ${OUT_AUDIO_TFRECORD_PATH}/train
rm -f FAILED

PROCESSORS_IN_PARALLEL=4
NUM_PROCESSORS=16
TOTAL_SHARDS=128
SHARD_PER_PROCESS=$((TOTAL_SHARDS / NUM_PROCESSORS))
LOOP=$((NUM_PROCESSORS / PROCESSORS_IN_PARALLEL))

for loopid in $(seq 1 ${LOOP}); do
    start=$(($((loopid - 1)) * ${PROCESSORS_IN_PARALLEL}))
    end=$(($start + PROCESSORS_IN_PARALLEL - 1))
    echo $start, $end
    for procid in $(seq $start $end); do
        set -x
        nice -n 10 python3 -m bytedseq.cli.create_tfrecords \
            --processor_id $procid --num_processors $NUM_PROCESSORS \
            --num_output_shards $TOTAL_SHARDS \
            --output_range_begin "$((SHARD_PER_PROCESS * procid))" \
            --output_range_end "$((SHARD_PER_PROCESS * procid + SHARD_PER_PROCESS))" \
        --dataset MuSTC --extraction "train" --add_language_tag\
        --feature_extractor.class floatidentity \
        --input_tarball ${INPUT_TARBALL} \
        --output_template ${OUT_AUDIO_TFRECORD_PATH}/train/train.tfrecords-%5.5d-of-%5.5d || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done


makeDirs ${OUT_AUDIO_TFRECORD_PATH}/devtest
for subset in dev tst-COMMON; do
    set -x
    nice -n 10 python3 -m bytedseq.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
    --dataset MuSTC --extraction ${subset} --add_language_tag \
    --feature_extractor.class floatidentity \
    --input_tarball ${INPUT_TARBALL} \
    --output_template ${OUT_AUDIO_TFRECORD_PATH}/devtest/${subset}.en-${TGT_LANG}.tfrecords-%5.5d-of-%5.5d || touch FAILED &
    set +x
done
wait
! [[ -f FAILED ]]
