CUDA_VISIBLE_DEVICES=0 \
python test.py \
--camera realsense \
--dataset_root "/home/yaohuayang/Dataset/dataset-data" \
--dump_dir /home/yaohuayang/adjustOffset/graspness_D_offset_cat/logs/realsense/test_10 \
--checkpoint_path  "/home/yaohuayang/adjustOffset/graspness_D_offset_cat/logs/realsense/minkuresunet_epoch10.tar" \
--infer \
--eval