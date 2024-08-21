import os 
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.DataManager import DataManager as DM
import re

class FilterDataset():
    @staticmethod
    def build_folder_pairs(target_path, path_rol="C:/Cours-Sorbonne/M1/Stage/src/data/rol_compressed", dest_path="C:/Cours-Sorbonne/M1/Stage/src/data/data_pairs"):
        """
        Build the folder pairs for the dataset
        Args:
        -----
            target_path (str): Path to the targets.npy file
            path_rol (str): Path to the rol dataset
            dest_path (str): Path to the destination folder
        """

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

            # Because sometimes the images have different width and height (1024,1023) for example
            img = Image.fromarray(DM.add_border_to_match_shape(np.array(img), shape=(1024,1024)))
            target = Image.fromarray(DM.add_border_to_match_shape(np.array(target), shape=(1024,1024))) 

            try : 
                ark = re.findall("bpt.*", target_image)[0]
                img.save(f"{dest_path}/pairs_{i}/{image}.jpg")
                target.save(f"{dest_path}/pairs_{i}/{ark}")
            except:
                continue


