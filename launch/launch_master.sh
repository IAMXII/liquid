#!/bin/bash
source ./launch/config.env

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=0
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

deepspeed --num_gpus=${GPUS_PER_NODE} ${TRAIN_FILE} ${RUN_ARGS}
