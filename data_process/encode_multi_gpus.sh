#!/bin/bash

INPUT_PAIRS=/data/bench2drive
VQGAN_PATH=./models/vqgan_imagenet_f16_1024
CACHE_ROOT=/data/gancache
NUM_CHUNKS=8
BATCH_SIZE=64
NUM_PROCESSES=8

mkdir -p logs

for IDX in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$IDX \
    python vq_encode_batch.py \
        --input_pairs ${INPUT_PAIRS} \
        --vqgan_path ${VQGAN_PATH} \
        --cache_root ${CACHE_ROOT} \
        --chunk_idx ${IDX} \
        --num_chunks ${NUM_CHUNKS} \
        --batch_size ${BATCH_SIZE} \
        --num_processes ${NUM_PROCESSES} \
        > logs/vq_encode_${IDX}.log 2>&1 &
done

wait
