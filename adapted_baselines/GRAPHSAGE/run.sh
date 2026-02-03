python GraphSAGE_train.py --dataset_name FILT_HB/edges1.csv
python GraphSAGE_train.py --dataset_name FILT_HB/edges2.csv

python postprocess.py \
  --file1_pattern temp_result/outputs_FILT_HB_edges1_seed{}.pt \
  --file2_pattern temp_result/outputs_FILT_HB_edges2_seed{}.pt \
  --summary1_pattern temp_result/summary_FILT_HB_edges1_seed{}.json \
  --summary2_pattern temp_result/summary_FILT_HB_edges2_seed{}.json