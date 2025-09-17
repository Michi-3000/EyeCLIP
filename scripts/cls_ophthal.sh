#!/bin/bash
current_time=$(date +"%Y-%m-%d-%H%M")
FINETUNE_CHECKPOINT="eyeclip_visual.pt"
DATA_PATH="/data/public/"

DATASET_NAMES=("IDRiD" "Aptos2019" "Glaucoma_Fundus" "JSIEC" "Retina" "MESSIDOR2" "OCTID" "PAPILA")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    echo "=============================================="
    echo "Processing dataset: $DATASET_NAME"
    echo "=============================================="
    
    python main_finetune_ophthal.py \
        --data_path "${DATA_PATH}" \
        --data_name "${DATASET_NAME}" \
        --finetune "${FINETUNE_CHECKPOINT}" \
        --clip_model_type "ViT-B/32" \
        --batch_size 64 \
        --epochs 50 \
        --lr 1e-4 \
        --weight_decay 0.01 \
        --output_dir "./classification_results/${current_time}" \
        --warmup_epochs 5 \
        --test_num 5
        
    echo "Finished processing dataset: $DATASET_NAME"
    echo ""
done

echo "All datasets processed successfully!"
