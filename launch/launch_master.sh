#!/bin/bash

# 先定义一个环境配置脚本，写到临时文件
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

# 给脚本执行权限
chmod +x setup_env.sh

# 批量 ssh 到所有节点执行 setup_env.sh
NODES=("10.54.99.213" "10.54.108.153")
echo "开始批量执行环境配置脚本..."
for node in ${NODES}; do
  echo "配置节点: $node"
  # 这里用 scp 先传脚本过去，再 ssh 执行
  scp setup_env.sh $node:~/setup_env.sh
  ssh $node 'bash ~/setup_env.sh'
done

echo "所有节点环境配置完成。"

# 删除临时脚本文件（如果不需要保留）
rm setup_env.sh

# 下面是你现有的 deepspeed 启动命令
source ./launch/config.env
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=0
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
#export CUDA_LAUNCH_BLOCKING=1
echo ${GPUS_PER_NODE}
deepspeed --hostfile hostfile.txt --num_gpus=${GPUS_PER_NODE} --num_nodes=${NUM_NODES} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  ${TRAIN_FILE} ${RUN_ARGS}
