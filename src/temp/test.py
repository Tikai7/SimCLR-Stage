import os 
import numpy as np 


def _get_pairs(path):
    pairs = None
    with open(path,"r") as f :
        pairs = f.readlines()
    pairs = [x.replace('/n','') for x in pairs]
    return pairs

def _remove_pairs(images, targets, pairs_to_remove):
    filtered_images = []
    filtered_targets = []
    for x, y in zip(images, targets):
        if x not in pairs_to_remove:
            filtered_images.append(x)
            filtered_targets.append(y)
    return filtered_images, filtered_targets

def _get_test_files(path_to_sim_test):
    temp_all_files = os.listdir(path_to_sim_test)
    temp_all_files = sorted(temp_all_files, key=lambda x:int(x.split("ID_")[1].split('.')[0]))
    temp_image_names = []
    for i in range(0,len(temp_all_files)-1,2):
        target = temp_all_files[i]
        img = temp_all_files[i+1]

        if not img.startswith("btv"):
            img, target = target, img
        temp_image_names.append(img.split("_ID")[0])

    return temp_image_names

bad_pairs_path = "C:/Cours-Sorbonne/M1/Stage/src/files/bad_pairs.txt"
to_enhance_path = "C:/Cours-Sorbonne/M1/Stage/src/files/to_enhance_pairs.txt"
target_path = "C:/Cours-Sorbonne/M1/Stage/src/data/rol_sim_rol_triplets/targets.npy"
path_sim_rol_test = "C:/Cours-Sorbonne/M1/Stage/src/data/data_PPTI/sim_rol_test"
caption_path = "C:/Cours-Sorbonne/M1/Stage/src/data/data_PPTI/sim_rol_super_compressed/captions"
target_names = np.load(target_path, allow_pickle=True).tolist()
images_names = np.load(target_path.replace("targets.npy","images.npy"), allow_pickle=True).tolist()

bad_pairs = _get_pairs(bad_pairs_path)
to_enhance_pairs = _get_pairs(to_enhance_path)

# Step 1: Remove None values
filtered_images = []
filtered_targets = []

for x, y in zip (images_names, target_names):
    if x is not None and y is not None:
        filtered_images.append(x)
        filtered_targets.append(y)



filtered_images, filtered_targets = _remove_pairs(filtered_images, filtered_targets, to_enhance_pairs)
filtered_images, filtered_targets = _remove_pairs(filtered_images, filtered_targets, bad_pairs)
filtered_images, filtered_targets = _remove_pairs(filtered_images, filtered_targets, _get_test_files(path_to_sim_test=path_sim_rol_test))

images_names = filtered_images
target_names = filtered_targets

import re 

captions = os.listdir(caption_path)

for y in target_names:
    try:
        ark = re.findall("bpt.*",y)[0].replace(".jpg",".txt")
        if ark in captions:
            with open(f"{caption_path}/{ark}","r") as f:
                y = f.readlines()[0]
                with open(f"C:/Cours-Sorbonne/M1/Stage/src/captions/{ark}","w") as f:
                    f.write(y)
    except:
        continue


