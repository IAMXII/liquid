#!/bin/bash


deepspeed --master_port 2935  --num_gpus 8 liquid/train/train_mem.py  \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /path/to/save/gemma-7b-addtoken \
    --version gemma \
    --data_path  /path/to/DCLM/hf_data/^^/path/to/JourneyDB/hf_data/   \
    --shuffleseed 42 \
    --percentage  0.5^^1.0  \
    --T2I_ratio 0.1 \
    --vq_resolution 512 \
    --image_folder ./data/ \
    --bf16 True \
    --lora_enable True \
    --output_dir  ./debug_gemma2b_mixpretrain_stage1 \
    --run_name  'debug_gemma2b_mixpretrain_stage1' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb



