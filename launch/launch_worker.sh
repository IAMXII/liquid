#!/bin/bash

# 从节点只需设置环境，deepspeed 会远程启动训练进程
source ~/anaconda3/etc/profile.d/conda.sh
conda activate liquid
export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export GIT_SSL_NO_VERIFY=true

cd /root/paddlejob/workspace/env_run/liuwei/liquid
source ./launch/config.env

echo "从节点已完成环境配置，等待 deepspeed 主节点连接..."
# 注意：无需执行任何训练命令，等待主节点启动后自动被拉起
sleep infinity
