#!/bin/bash

# 临时写入 setup_env.sh，用于激活 conda 和设置代理等
cat > setup_env.sh <<'EOF'
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate liquid
export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export GIT_SSL_NO_VERIFY=true
cd /root/paddlejob/workspace/env_run/liuwei/liquid
source ./launch/config.env
EOF

chmod +x setup_env.sh

# 配置两个节点（示例 IP，替换成你自己的）
NODES=("10.54.108.153" "10.54.99.213")

echo "开始批量执行环境配置脚本..."
for node in "${NODES[@]}"; do
  echo "配置节点: $node"
  scp setup_env.sh $node:~/setup_env.sh
  ssh $node 'bash ~/setup_env.sh'
done

rm setup_env.sh
echo "所有节点环境配置完成。"

# ---------------------- torchrun 启动命令 ----------------------

# 通用配置
NUM_NODES=2
GPUS_PER_NODE=8
MASTER_ADDR="10.54.99.213"   # 通常是 node0 的 IP
MASTER_PORT=29500
TRAIN_SCRIPT=${TRAIN_FILE}   # 比如 train.py
RUN_ARGS=${RUN_ARGS}         # 你额外的训练参数，比如 --batch_size 64 等

# 节点 0 上启动
if [[ "$(hostname -I)" =~ "$MASTER_ADDR" ]]; then
  echo "当前为主节点，开始 torchrun 启动..."

  torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT $RUN_ARGS
else
  echo "当前为从节点，监听 torchrun 启动..."
  torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT $RUN_ARGS
fi
