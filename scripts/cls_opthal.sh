current_time=$(date +"%Y-%m-%d-%H%M")
for epoch in "eyeclip"; do
  checkpoint="checkpoint-${epoch}.pth"
  for name in 'IDRiD' 'OCTID' 'PAPILA' 'Retina' 'JSIEC' 'MESSIDOR2' 'Aptos2019' 'Glaucoma_Fundus' 'OCTDL' 'Retina Image Bank'; do
    CUDA_VISIBLE_DEVICES=2 python main_finetune_opthal.py \
      --now_epoch $epoch \
      --test_num 5 \
      --data_name $name \
      --batch_size 16 \
      --world_size 1 \
      --model vit_large_patch16 \
      --epochs 50 \
      --blr 5e-3 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.2 \
      --output_dir "output_dir_downstream/all_dataset_$current_time" \
      --data_path "" \
      --finetune "$checkpoint" \
      --input_size 224
  done
done


