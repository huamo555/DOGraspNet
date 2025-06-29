import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/yaohuayang/Dataset/dataset-data/')

parser.add_argument('--dump_dir', help='Dump dir to save outputs', default="/home/yaohuayang/adjustOffset/graspness_D_offset_cat/logs/realsense/test_10/")

parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 15000]')

parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')

parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
# 设置视角
parser.add_argument('--index', type=str, default='0000')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
    color = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')), dtype=np.float32) / 255.0
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')))
    meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    return ret_dict,cloud


if __name__ == '__main__':
    

    for i in range(100, 188, 1):
        
        scene_id = 'scene_' + str(i).zfill(4)
        index = cfgs.index
        data_dict,cloud = data_process()

        
        #pc = data_dict['point_clouds']
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 10:
            gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        
        for gripper in grippers:
            if isinstance(gripper, o3d.geometry.TriangleMesh):
                point_cloud = gripper.sample_points_uniformly(number_of_points=20000)
                cloud += point_cloud
            elif isinstance(gripper, o3d.geometry.PointCloud):
                cloud += gripper
        if not os.path.exists("./vis/"+cfgs.camera+"/"):
            os.makedirs("./vis/"+cfgs.camera+"/")
        o3d.io.write_point_cloud("./vis/"+cfgs.camera+"/"+scene_id+"-"+str(i).zfill(4)+".pcd", cloud)
        print("保存成功")        

        
