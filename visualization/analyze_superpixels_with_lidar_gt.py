import numpy as np
import pickle
import os
from PIL import Image
from tqdm import tqdm

# define paths
nuscenes_version = 'v1.0-mini'
# nuscenes_version = 'v1.0-trainval'

# stats file path
stats_path_root = "../superpixels_mini/"
# stats_path_root = "../superpixels/"


if nuscenes_version == 'v1.0-mini':
    superpixel_path = "../superpixels/nuscenes/superpixels_slic_mini/"
    lidar_mask_path = "../superpixels/nuscenes/lidar_mask_mini/"
else:
    superpixel_path = "../superpixels/nuscenes/superpixels_slic/"
    lidar_mask_path = "../superpixels/nuscenes/lidar_mask/"


if not os.path.isdir(lidar_mask_path):
    print('lidar masks not found!')

# get the list of lidar mask files
lidar_mask_list = os.listdir(lidar_mask_path)


superpixel_dict = {'total': 0}
class_info = {}
superpixel_cnt = {}

# construct a dict for storing general superpixel stats
# each key is the diversity count for superpixels
# the format for each value: [temporary count, sum of percentage of lidar point from dominant class]
for i in range(100):
    superpixel_cnt[i] = [0, 0]

# construct a dict for storing stats for each class
# each key is the class number from nuscenes except for noise being 100 instead of 0 to differentiate between background pixels and noise pixels
# the format for each value: [total number of superpixels dominated by a class, sum of percentage of lidar point from the class 
#                             ,sum of how many different class of point cloud appear in a superpixel dominated by this class]
for i in range(32):
    class_info[i] = [0, 0, 0]
class_info[100] = [0, 0, 0]


for file in tqdm(lidar_mask_list):
    superpixels = np.asarray(Image.open(superpixel_path + file))
    lidar_mask = np.asarray(Image.open(lidar_mask_path + file))

    # get the ids of all superpixels
    ids = np.unique(superpixels)

    # update the total superpixels count
    superpixel_dict['total'] += len(ids)

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

        # update dicts
        superpixel_cnt[superpixel_diversity_count][0] += 1
        superpixel_cnt[superpixel_diversity_count][1] += (dominant_num_cnt / sum_cnt)
        
        class_info[tmp_dominant_class][0] += 1
        class_info[tmp_dominant_class][1] += (dominant_num_cnt / sum_cnt)
        class_info[tmp_dominant_class][2] += superpixel_diversity_count

# normalize the results to percentages and save the results
results = {'class_info': {}, 'superpixel_stats': {}, 'total': superpixel_dict}
for k in superpixel_cnt.keys():
    superpixel_info = superpixel_cnt[k]
    actual_result1 = [superpixel_info[0] / (superpixel_dict['total'] + np.finfo(float).eps), superpixel_info[1] / (superpixel_info[0] + np.finfo(float).eps)]
    results['superpixel_stats'][k] = actual_result1

for k in class_info.keys():
    per_class_info = class_info[k]
    actual_result2 = [per_class_info[0] / (superpixel_dict['total'] + np.finfo(float).eps), per_class_info[1] / (per_class_info[0] + np.finfo(float).eps), per_class_info[2] / (per_class_info[0] + np.finfo(float).eps)]
    results['class_info'][k] = actual_result2

if not os.path.isdir(stats_path_root):
    os.makedirs(stats_path_root)

with open(stats_path_root+ "lidar_stats.pkl", "wb") as f:
    pickle.dump(results, f)

