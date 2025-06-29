CUDA_VISIBLE_DEVICES=0 \
python test.py \
--camera kinect \
--dataset_root /home/yaohuayang/Dataset/dataset-data/ \
--dump_dir /home/yaohuayang/adjustOffset/graspness_D_offset_cat/logs/kinect/test_10 \
--checkpoint_path  "/home/yaohuayang/adjustOffset/graspness_D_offset_cat/logs/kinect/minkuresunet_epoch10.tar" \
--infer \
--eval