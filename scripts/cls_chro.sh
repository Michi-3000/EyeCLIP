current_time=$(date +"%Y-%m-%d-%H%M")
for epoch in "eyeclip"
do
  checkpoint="checkpoint-${epoch}.pth"
  CUDA_VISIBLE_DEVICES=0 python main_finetune_chro.py \
    --now_epoch $epoch \
    --test_num 5 \
    --data_name chro \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path data/public \
    --output_dir "output_dir_downstream/chronicdisease5/$current_time" \
    --finetune "$checkpoint" \
    --input_size 224
done
