import os 
import numpy as np
from tqdm import tqdm
from PIL import Image
import re

class FilterDataset():
    @staticmethod
    def build_folder_pairs(target_path, path_rol="C:/Cours-Sorbonne/M1/Stage/src/data/rol_compressed" ,dest_path="C:/Cours-Sorbonne/M1/Stage/src/data/data_pairs"):
        
        images_names = np.load(target_path.replace("targets.npy","images.npy"), allow_pickle=True).tolist()
        target_names = np.load(target_path, allow_pickle=True).tolist()

        for x,y in zip(images_names.copy(), target_names.copy()):
            if x is None or y is None :
                images_names.remove(x)
                target_names.remove(y)

        for i, (image,target_image) in tqdm(enumerate(zip(images_names, target_names))):
            os.makedirs(f"{dest_path}/pairs_{i}")
            image_file = image.split('.')[0]
            img = Image.open(os.path.join(path_rol,image_file)+".jpg").convert('RGB')
            target = Image.open(target_image.replace("\\","/")).convert('RGB')
            try : 
                ark = re.findall("bpt.*", target_image)[0]
                img.save(f"{dest_path}/pairs_{i}/{image}.jpg")
                target.save(f"{dest_path}/pairs_{i}/{ark}")
            except:
                continue
