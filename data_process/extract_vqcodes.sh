#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python convert_imagepair_cc512.py \
        --input_pairs /path/to/valid_pairs.jsonl  \
        --temp_path /path/to/JourneyDB/temp  \
        --save_path /path/to/JourneyDB/hf_data  \
        --vqgan_path /path/to/chameleon/VQGAN/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done
wait


python packing_imagepairs.py  --temp_path /path/to/JourneyDB/temp  --save_path /path/to/JourneyDB/hf_data 
   