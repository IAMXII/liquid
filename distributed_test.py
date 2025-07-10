# distributed_test.py
import os
import socket
import torch
import torch.distributed as dist


def main():
    # 初始化分布式进程组，nccl后端仅支持GPU
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    print(f"[Rank {rank}/{world_size}] running on {hostname}")

    # 简单同步 barrier，确保所有进程都运行到这里
    dist.barrier()

    if rank == 0:
        print("All processes synchronized. Distributed setup works!")


if __name__ == '__main__':
    main()
