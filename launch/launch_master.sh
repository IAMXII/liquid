#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate liquid
export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export GIT_SSL_NO_VERIFY=true
source ./launch/config.env

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=0
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
#export CUDA_LAUNCH_BLOCKING=1

#deepspeed --num_gpus=${GPUS_PER_NODE} ${TRAIN_FILE} ${RUN_ARGS}
deepspeed --hostfile hostfile.txt --num_gpus=${GPUS_PER_NODE} --num_nodes=${NUM_NODES} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  ${TRAIN_FILE} ${RUN_ARGS}
