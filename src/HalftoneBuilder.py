import os
import cv2
from tqdm import tqdm
from utils.Processing import Processing as PC

path_rol_comp = "../data/rol_super_compressed" 
path_sim_rol_extracted_comp = "../data/sim_rol_super_compressed" 
path_filtered = "../data/rol_super_compressed/json_filtered"
path_targets = "../data/rol_sim_rol_couples/targets.npy"
bad_pairs_path = "./files/bad_pairs.txt"
to_enhance_path = "./files/to_enhance_pairs.txt"
path_rol_ht = "../data/rol_ht_super_compressed"

all_computed_files = set(os.listdir(path_rol_ht))

for i, file in tqdm(enumerate(os.listdir(path_rol_comp))):
    try:
        if file not in all_computed_files:
            img = cv2.imread(path_rol_comp+"/"+file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img_pil = PC.to_halftone(img)
            img_pil.save(path_rol_ht+"/"+file)
    except:
        continue


