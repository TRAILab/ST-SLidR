import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import cv2

class_mapping = {0: 'background',
                1: 'animal',
                2: 'human.pedestrian.adult',
                3: 'human.pedestrian.child',
                4: 'human.pedestrian.construction_worker',
                5: 'human.pedestrian.personal_mobility',
                6: 'human.pedestrian.police_officer',
                7: 'human.pedestrian.stroller',
                8: 'human.pedestrian.wheelchair',
                9: 'movable_object.barrier',
                10: 'movable_object.debris',
                11: 'movable_object.pushable_pullable',
                12: 'movable_object.trafficcone',
                13: 'static_object.bicycle_rack',
                14: 'vehicle.bicycle',
                15: 'vehicle.bus.bendy',
                16: 'vehicle.bus.rigid',
                17: 'vehicle.car',
                18: 'vehicle.construction',
                19: 'vehicle.emergency.ambulance',
                20: 'vehicle.emergency.police',
                21: 'vehicle.motorcycle',
                22: 'vehicle.trailer',
                23: 'vehicle.truck',
                24: 'flat.driveable_surface',
                25: 'flat.other',
                26: 'flat.sidewalk',
                27: 'flat.terrain',
                28: 'static.manmade',
                29: 'static.other',
                30: 'static.vegetation',
                31: 'vehicle.ego',
                100: 'noise'}

# define dataset
nuscenes_version = 'v1.0-mini'
# nuscenes_version = 'v1.0-trainval'


if nuscenes_version == 'v1.0-mini':
    superpixel_path = "../superpixels/nuscenes/superpixels_slic_mini/"
    lidar_mask_path = "../superpixels/nuscenes/lidar_mask_mini/"
    generated_masks_root = "../superpixels/nuscenes/foreground_masks_mini/"
else:
    superpixel_path = "../superpixels/nuscenes/superpixels_slic/"
    lidar_mask_path = "../superpixels/nuscenes/lidar_mask/"
    generated_masks_root = "../superpixels/nuscenes/foreground_masks/"

if not os.path.isdir(lidar_mask_path):
    print('lidar masks not found!')

# get the list of lidar mask files
lidar_mask_list = os.listdir(lidar_mask_path)

if not os.path.isdir(generated_masks_root):
    os.makedirs(generated_masks_root)


for file in tqdm(lidar_mask_list):
    superpixels = np.asarray(Image.open(superpixel_path + file))
    lidar_mask = np.asarray(Image.open(lidar_mask_path + file))
    # get the ids of all superpixels
    ids = np.unique(superpixels)

    # initialize foreground masks
    foreground_mask = np.zeros(lidar_mask.shape).astype(np.uint8)

    for id in ids:
        sum_cnt = 0

        values, counts = np.unique(lidar_mask[superpixels==id], return_counts=True)
        dominant_num_cnt = 0
        tmp_dominant_class = 0
        superpixel_diversity_count = 0

        # find the dominant class and its count
        for i in range(len(values)):
            if values[i] == 0:
                continue
            else:
                sum_cnt += counts[i]
                superpixel_diversity_count += 1
                if counts[i] > dominant_num_cnt:
                    dominant_num_cnt = counts[i]
                    tmp_dominant_class = values[i]

        # handle superpixels with only background pixels
        if dominant_num_cnt == 0 and tmp_dominant_class == 0:
            for j in range(len(values)):
                if values[j] == 0:
                    dominant_num_cnt = counts[j]
                    sum_cnt = dominant_num_cnt

        # make background, driveavle_surfacem sidewalk, terrain, manmade, vegegation, noise as background superpixels
        if tmp_dominant_class == 0 or tmp_dominant_class == 24 or tmp_dominant_class == 26 or \
            tmp_dominant_class == 27 or tmp_dominant_class == 28 or tmp_dominant_class == 30 or\
            tmp_dominant_class == 100:
            continue

        foreground_mask[superpixels==id] = 1
    
    cv2.imwrite(generated_masks_root + file, foreground_mask)
