#!/bin/bash
RANK_ID=$1
if [ -z "$RANK_ID" ]; then
  echo "Usage: bash launch_train_worker.sh <node_rank>"
  exit 1
fi

source ./launch/config.env

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=$((RANK_ID * GPUS_PER_NODE))
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
#export CUDA_LAUNCH_BLOCKING=1

deepspeed --num_gpus=${GPUS_PER_NODE} ${TRAIN_FILE} ${RUN_ARGS}
#deepspeed --hostfile hostfile.txt --num_gpus=${GPUS_PER_NODE} --num_nodes=${NUM_NODES} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  ${TRAIN_FILE} ${RUN_ARGS}
