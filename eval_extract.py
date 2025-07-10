import os
import random
import shutil

# 路径配置
bench2drive_root = "/data/bench2drive"  # 原始数据集
val_dir = "/data/bench2drive_val"  # 验证集目标路径
val_output_file = "./val_split.txt"  # 保存验证集名称
val_size = 100  # 验证集大小
random_seed = 42  # 保证复现

# 设置随机种子
random.seed(random_seed)

# 获取所有 sequence 子目录
sequences = sorted([
    seq for seq in os.listdir(bench2drive_root)
    if os.path.isdir(os.path.join(bench2drive_root, seq))
])

print(f"总共找到 {len(sequences)} 个 sequences")

# 随机抽取验证集 sequence
val_seqs = random.sample(sequences, val_size)

# 写入验证集列表文件
with open(val_output_file, "w") as f:
    for seq in val_seqs:
        f.write(seq + "\n")

print(f"✅ 已保存验证集列表到 {val_output_file}")

# 创建验证集目录
os.makedirs(val_dir, exist_ok=True)
vqcache_root = "/data/vqcache"
vqcache_eval = "/data/vqcache_val"
# 复制并删除原始目录中的验证集 sequence
for seq in val_seqs:
    src = os.path.join(bench2drive_root, seq)
    dst = os.path.join(val_dir, seq)
    print(f"复制 {seq} 到验证集目录...")
    src1 = os.path.join(vqcache_root, seq)
    dst1 = os.path.join(vqcache_eval, seq)
    shutil.copytree(src, dst)
    shutil.copytree(src1, dst1)

    print(f"删除原始目录中的 {seq} ...")
    shutil.rmtree(src)
    shutil.rmtree(src1)

print(f"✅ 已完成验证集迁移和原始数据清理")
