#CUDA_VISIBLE_DEVICES=4 python train.py --camera kinect --log_dir logs/log_kn --batch_size 4 --learning_rate 0.001 --model_name minkuresunet --dataset_root /data3/graspnet

#  --model_name minkuresunet --dataset_root /data3/yaohuayang/Dataset/dataset-data/ --resume --checkpoint_path ./logs/log_kn/minkuresunet_epoch07.tar

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--log_dir "logs/realsense" \
--dataset_root /home/yaohuayang/Dataset/dataset-data/ \
--camera "realsense" \
--batch_size "2" \
--accum_iter "2" \
--model_name "minkuresunet"









