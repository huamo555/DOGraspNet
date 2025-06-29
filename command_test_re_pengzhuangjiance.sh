CUDA_VISIBLE_DEVICES=0 \
python test.py \
--camera realsense \
--dataset_root /data_1/yaohuayang_data/Dataset/dataset-data/ \
--dump_dir /data_1/yaohuayang_data/adjustOffset/graspness_D_offset_cat/logs/realsense/test_10_pengzhuangjiance \
--checkpoint_path  "/data_1/yaohuayang_data/adjustOffset/graspness_D_offset_cat/logs/realsense/minkuresunet_epoch10.tar" \
--infer \
--eval \
--collision_thresh 0.01