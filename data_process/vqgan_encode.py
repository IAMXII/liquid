# vq_encode_batch.py (Multi-GPU enabled)
import os, json, argparse
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import numpy as np
from multiprocessing.pool import ThreadPool
from vqgan.image_tokenizer import ImageTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def center_crop(image, size=512):
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def collect_images(root):
    all_paths = []
    for seq in sorted(os.listdir(root)):
        img_dir = os.path.join(root, seq, 'camera/rgb_front')
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.jpg'):
                all_paths.append((seq, os.path.join(img_dir, fname)))
    return all_paths


def process_batch(batch_paths, image_tokenizer, cache_root, seq_name):
    images = []
    output_paths = []
    for _, img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            if max(img.size) / min(img.size) > 2:
                continue
            img = center_crop(img)
            images.append(img)
            frame_name = os.path.basename(img_path).replace('.jpg', '.npy')
            output_paths.append(os.path.join(cache_root, seq_name, frame_name))
        except Exception as e:
            print(f"Image load/crop error: {img_path}, {e}")
            continue

    if not images:
        return

    with torch.no_grad():
        for img, out_path in zip(images, output_paths):
            try:
                vqcode = image_tokenizer.img_tokens_from_pil(img)
                vqcode = vqcode.cpu().numpy()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, vqcode)
            except Exception as e:
                print(f"VQ encode failed: {out_path}, {e}")
                continue


def main(args):
    # 自动获取当前可见的 GPU ID（假设用 CUDA_VISIBLE_DEVICES 限制）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_rank = args.chunk_idx if args.chunk_idx < len(visible_devices) else 0
        device = f"cuda:{local_rank}"
    print(f"Using device: {device} for chunk {args.chunk_idx}")

    print(f"Loading VQGAN tokenizer from {args.vqgan_path}")
    vqgan_cfg_path = os.path.join(args.vqgan_path, "configs", "model.yaml")
    vqgan_ckpt_path = os.path.join(args.vqgan_path, "ckpts", "last.ckpt")

    image_tokenizer = ImageTokenizer(
        cfg_path=vqgan_cfg_path,
        ckpt_path=vqgan_ckpt_path,
        device=device
    )

    all_images = collect_images(args.input_pairs)
    chunks = np.array_split(all_images, args.num_chunks)[args.chunk_idx]
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

    print(f"Processing {len(chunks)} images in chunk {args.chunk_idx}")
    pbar = tqdm(total=total_batches, desc=f"Chunk {args.chunk_idx}")

    pool = ThreadPool(processes=args.num_processes)

    def thread_job(batch):
        if len(batch) == 0:
            return
        seq_name = batch[0][0]
        process_batch(batch, image_tokenizer, args.cache_root, seq_name)
        pbar.update(1)

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        pool.apply_async(thread_job, args=(batch,))

    pool.close()
    pool.join()
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pairs', type=str, required=True)
    parser.add_argument('--vqgan_path', type=str, required=True)
    parser.add_argument('--cache_root', type=str, default='/data/vqcache')
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()
    main(args)
