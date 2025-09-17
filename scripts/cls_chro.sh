current_time=$(date +"%Y-%m-%d-%H%M")
FINETUNE_CHECKPOINT="eyeclip_visual.pt"

CUDA_VISIBLE_DEVICES=0 python main_finetune_chro.py \
    --finetune "${FINETUNE_CHECKPOINT}" \
    --clip_model_type "ViT-B/32" \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --output_dir "./classification_results/${current_time}" \
    --warmup_epochs 5 \
    --test_num 5
