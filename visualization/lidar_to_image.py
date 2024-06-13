from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors


# define path to nuscenes dataset and superpixels

nuscenes_version = 'v1.0-mini'
# nuscenes_version = 'v1.0-trainval'

nuscenes_path = "../datasets/nuscenes"

if nuscenes_version == 'v1.0-mini':
    print('use nuscenes mini')
    superpixels_path = "../superpixels/nuscenes/superpixels_slic_mini/"
    lidar_masks_path = "../superpixels/nuscenes/lidar_mask_mini/"    
else:
    print('use nuscenes')
    superpixels_path = "../superpixels/nuscenes/superpixels_slic/"
    lidar_masks_path = "../superpixels/nuscenes/lidar_mask/"

camera_list = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


if not os.path.isdir(lidar_masks_path):
    os.makedirs(lidar_masks_path)

nusc = NuScenes(
    version=nuscenes_version, dataroot=nuscenes_path, verbose=False
)
print('start processing')

# get nuscenes color legend
color_legend = colormap_to_colors(nusc.colormap, nusc.lidarseg_name2idx_mapping)

# iterate over all samples
for scene_idx in tqdm(range(len(nusc.scene))):
    scene = nusc.scene[scene_idx]
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "":
        current_sample = nusc.get("sample", current_sample_token)
        for camera_name in camera_list:
            cam = nusc.get("sample_data", current_sample["data"][camera_name])
            superpixel_path = superpixels_path + cam["token"] + ".png"

            assert os.path.exists(superpixel_path) == True
            superpixel = np.asarray(Image.open(superpixel_path))

            # get point cloud and coloring labels
            points, coloring, _ = nusc.explorer.map_pointcloud_to_image(current_sample["data"]["LIDAR_TOP"], 
                                                                current_sample["data"][camera_name],
                                                                show_lidarseg=True)

            # convert to 2d coordinates
            img_points = points[:2,:].astype(np.int64)

            # construct lidar mask
            # set the value at projected pixel to be the class number
            # background pixels are zeros, the noise pixels labeled by nuscenes is set to 100 to differentiate between background and noise
            lidar_mask = np.zeros(superpixel.shape)
            for i in range(len(coloring)):
                for j in range(len(color_legend)):
                    if np.array_equal(coloring[i], color_legend[j]):
                        if j == 0:
                            lidar_mask[img_points[1,i],img_points[0,i]] = 100
                        else:
                            lidar_mask[img_points[1,i],img_points[0,i]] = j
                        break
            
            # save lidar mask
            lidar_mask = lidar_mask.astype(np.uint8)
            cv2.imwrite(lidar_masks_path + cam["token"] + ".png", lidar_mask)

        current_sample_token = current_sample["next"]