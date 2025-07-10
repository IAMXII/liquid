source ./launch/config.env
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=0
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "准备启动 DeepSpeed 多机训练..."
deepspeed   --num_gpus=${GPUS_PER_NODE} \
  ${TRAIN_FILE} ${RUN_ARGS}