python extra_preprocess.py DG_data/FILT_HB_DIST/edges1.csv DG_data/FILT_HB_DIST/FILT_HB_DIST.csv
python extra_preprocess.py DG_data/FILT_HB_ANGLE/edges2.csv DG_data/FILT_HB_ANGLE/FILT_HB_ANGLE.csv
cd preprocess_data/
python preprocess_data.py --dataset_name FILT_HB_DIST
python preprocess_data.py --dataset_name FILT_HB_ANGLE
cd ../
python train_link_prediction.py --dataset_name FILT_HB_DIST --model_name TGAT --optimizer RMSprop --num_layers 1 --batch_size 100 --num_neighbors 50 --num_runs 1 --dropout 0.1 --weight_decay 0.0001 --test_interval_epochs 1 --num_epochs 300
python train_link_prediction.py --dataset_name FILT_HB_ANGLE --model_name TGAT --optimizer RMSprop --num_layers 1 --batch_size 100 --num_neighbors 50 --num_runs 1 --dropout 0.1 --weight_decay 0.0001 --test_interval_epochs 1 --num_epochs 300
python postprocess.py --file1_pattern temp_result/outputs_FILT_HB_DIST_TGAT_seed{}.pt --file2_pattern temp_result/outputs_FILT_HB_ANGLE_TGAT_seed{}.pt --summary1_pattern temp_result/summary_FILT_HB_DIST_TGAT_seed{}.json --summary2_pattern temp_result/summary_FILT_HB_ANGLE_TGAT_seed{}.json