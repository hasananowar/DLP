conda activate skipgnn
cd SkipGNN

python train.py \
    --epochs 300 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --batch_size 100 \
    --dropout 0.1 \
    --hidden1 16 \
    --hidden2 8 \
    --hidden_decode1 4 \
    --network_type FILT_HB_DIST \
    --data_path '../data/FILT_HB_DIST/fold1' \
    --input_type one_hot
python train.py \
    --epochs 300 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --batch_size 100 \
    --dropout 0.1 \
    --hidden1 16 \
    --hidden2 8 \
    --hidden_decode1 4 \
    --network_type FILT_HB_ANGLE \
    --data_path '../data/FILT_HB_ANGLE/fold1' \
    --input_type one_hot

python postprocess.py \
  --file1_pattern temp_result/outputs_FILT_HB_DIST_seed{}.pt \
  --file2_pattern temp_result/outputs_FILT_HB_ANGLE_seed{}.pt \
  --summary1_pattern temp_result/summary_FILT_HB_DIST_seed{}.json \
  --summary2_pattern temp_result/summary_FILT_HB_ANGLE_seed{}.json