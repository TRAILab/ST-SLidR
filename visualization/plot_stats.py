import pickle
from matplotlib import pyplot as plt

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

# stats file path
stats_path_root = "../superpixels_mini/"
# stats_path_root = "../superpixels/"


stats_path = stats_path_root + "lidar_stats.pkl"

with open(stats_path, "rb") as f:
    stats = pickle.load(f)

class_info = stats['class_info']
superpixel_stats = stats['superpixel_stats']
print('total number of superpixels:', stats['total']['total'])

# handle class info
class_percentage, class_label, class_dominance, class_diversity = [], [], [], []
include_background = False
if include_background:
    for k in class_info.keys():
        class_label.append(class_mapping[k])
        class_percentage.append(class_info[k][0])
        class_dominance.append(class_info[k][1])
        class_diversity.append(class_info[k][2])
        # print(class_mapping[k], class_percentage)

else:
    background_percentage = class_info[0][0]
    others_percentage = 1 - background_percentage
    for k in class_info.keys():
        if k == 0:
            continue
        class_label.append(class_mapping[k])
        class_percentage.append(class_info[k][0] / others_percentage)
        class_dominance.append(class_info[k][1])
        class_diversity.append(class_info[k][2])
        # print(class_mapping[k], class_info[k][0] / others_percentage)


plt.figure(figsize=(20,20))
plt.barh(class_label,class_percentage,height=0.8)
plt.gca().invert_yaxis
plt.ylabel('Percentage')
plt.title('percentage of superpixels among all superpixels that are occupied by each class in nuScenes v1.0-trainval')
plt.tight_layout()
# plt.show()
plt.savefig(stats_path_root + "class_info_percentage_stats.png",dpi=250)

plt.figure(figsize=(20,20))
plt.barh(class_label,class_dominance,height=0.8)
plt.gca().invert_yaxis
plt.ylabel('Percentage')
plt.title('Average percentage of point cloud inside the superpixel that belongs to the dominant class in nuScenes v1.0-trainval')
plt.tight_layout()
# plt.show()
plt.savefig(stats_path_root + "class_info_dominance_stats.png",dpi=250)

plt.figure(figsize=(20,20))
plt.barh(class_label,class_diversity,height=0.8)
plt.gca().invert_yaxis
plt.ylabel('Count')
plt.title('Average number of different types of class (including itself) in a superpixel that is dominated by the given class in nuScenes v1.0-trainval')
plt.tight_layout()
# plt.show()
plt.savefig(stats_path_root + "class_info_diversity_stats.png",dpi=250)

# handle superpixel stats
superpixel_text, superpixel_diversity_percentage, superpixel_dominance_percentage = [], [], []
for k in range(16):
    superpixel_text.append(k)
    superpixel_diversity_percentage.append(superpixel_stats[k][0])
    superpixel_dominance_percentage.append(superpixel_stats[k][1])

plt.figure(figsize=(20,20))
plt.barh(superpixel_text,superpixel_diversity_percentage,height=0.8)
plt.gca().invert_yaxis
plt.ylabel('Percentage')
plt.title('Average percentage of superpixels that occupied by N different classes (excluding background and noise) in nuScenes v1.0-trainval')
plt.tight_layout()
# plt.show()
plt.savefig(stats_path_root + "superpixel_diversity_stats.png",dpi=250)

plt.figure(figsize=(20,20))
plt.barh(superpixel_text,superpixel_dominance_percentage,height=0.8)
plt.gca().invert_yaxis
plt.ylabel('Percentage')
plt.title('Average percentage of point cloud inside the superpixel that belongs to the dominant class in nuScenes v1.0-trainval')
plt.tight_layout()
# plt.show()
plt.savefig(stats_path_root + "superpixel_dominance_stats.png",dpi=250)
