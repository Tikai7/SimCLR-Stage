from tqdm import tqdm
import cv2
import os
import numpy as np
import glob
import torch
from utils.JSONRetriever import JSONRetriever as JR
from torch.utils.data import Dataset
from utils.Processing import RandomRotation90
from utils.Processing import Processing as PC 
from torchvision import transforms
from PIL import Image
from model.SIFT import SIFTDetector

import matplotlib.pyplot as plt

SSH = os.getcwd() != 'c:\\Cours-Sorbonne\\M1\\Stage\\src'

class DataLoaderSimCLR(Dataset):
    def __init__(
            self, path_rol, path_sim_rol_nn_extracted, path_json_filtered, 
            shape=(256,256),
            target_path="C:/Cours-Sorbonne/M1/Stage/src/data/rol_sim_rol_triplets/targets.npy", 
            bad_pairs_path = "C:/Cours-Sorbonne/M1/Stage/src/files/bad_pairs.txt", 
            to_enhance_path = "C:/Cours-Sorbonne/M1/Stage/src/files/to_enhance_pairs.txt",
            path_sim_rol_test = "C:/Cours-Sorbonne/M1/Stage/src/data/data_PPTI/sim_rol_test",
            path_to_halftone_images = None,
            augment_test=False, use_only_rol=False, build_if_error = False, max_images=None, use_context=False,
            remove_to_enhance_files=False, remove_bad_pairs=False, remove_sub_testset=True, encode_context=False
    ) -> None:

        self.path_ht_rol = path_to_halftone_images
        self.use_context = use_context
        self.use_only_rol = use_only_rol
        self.augment_test = augment_test
        self.path_filtered = path_json_filtered
        self.path_rol = path_rol
        self.path_sim_rol_nn_extracted = path_sim_rol_nn_extracted
        self.shape = shape
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        kernel_size = int(0.1 * self.shape[0]) if (int(0.1 * self.shape[0])%2 != 0) else int(0.1 * self.shape[0])-1
        self.transform_simclr = transforms.Compose([
                transforms.Resize(self.shape),  
                transforms.RandomApply([transforms.RandomResizedCrop(size=self.shape, scale=(0.2, 1.0))],p=0.5),
                transforms.RandomApply([transforms.Lambda(lambda x : PC.apply_rotogravure_effect(x,method="grid"))], p=0.5),
                transforms.RandomApply([RandomRotation90()], p=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.2, 0.5))], p=0.5)
        ]) 
        
        self.transform = transforms.Compose([
                transforms.Resize(self.shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        try:
            if self.use_only_rol  : 
                self.test_images = np.load(target_path.replace("targets.npy","images.npy"), allow_pickle=True).tolist()
                self.images_names = os.listdir(path_rol)[:max_images] if max_images is not None else os.listdir(path_rol)
                self.images_names = [x for x in self.images_names if "jpg" in x]
                self.images_names = [x for x in self.images_names if x.split('.')[0] not in self.test_images]
                print(f"[INFO] Using ROL Dataset with {len(self.images_names)} images")
            else:
                self.images_names = np.load(target_path.replace("targets.npy","images.npy"), allow_pickle=True).tolist()
                self.target_names = np.load(target_path, allow_pickle=True).tolist()
            print("[INFO] Loaded exsisting targets")
        except:
            print("[ERROR] Failed loading targets")
            print("[INFO] Creating targets...")
            self.triplets = JR.get_all_relations(path_json_filtered)[1]
            self.images_names = list(self.triplets.keys())
            self.target_names = list('_'.join(x[0].replace('/','_').replace("'}","").split('.')[0].split('_')[1:]) for x in self.triplets.values())
            sim_rol_files = set([x.split('_')[0] for x in os.listdir(path_sim_rol_nn_extracted)])
            for x,y in zip(self.images_names.copy(), self.target_names.copy()):
                if y.split('_')[0] not in sim_rol_files:
                    self.images_names.remove(x)
                    self.target_names.remove(y)
            
            images_path = target_path.replace("targets.npy","images.npy")
            if build_if_error : 
                np.save(images_path, self.images_names)
                self.build_dataset(target_path)        
            else:
                raise ImportError("Can import the dataset, please set build_if_error at 'True'")
        
        if not use_only_rol:
            print(f"[INFO] Before filtering : {len(self.images_names)} images")
            bad_pairs = self._get_pairs(bad_pairs_path)
            to_enhance_pairs = self._get_pairs(to_enhance_path)
            
            # Step 1: Remove None values
            filtered_images = []
            filtered_targets = []

            for x, y in zip(self.images_names, self.target_names):
                if x is not None and y is not None:
                    filtered_images.append(x)
                    filtered_targets.append(y)



            if remove_to_enhance_files:
                filtered_images, filtered_targets = self._remove_pairs(filtered_images, filtered_targets, to_enhance_pairs)

            if remove_bad_pairs : 
                filtered_images, filtered_targets = self._remove_pairs(filtered_images, filtered_targets, bad_pairs)

            if remove_sub_testset : 
                filtered_images, filtered_targets = self._remove_pairs(filtered_images, filtered_targets, self._get_test_files(path_to_sim_test=path_sim_rol_test))
            
            self.images_names = filtered_images
            self.target_names = filtered_targets

            print(f"[INFO] After filtering : {len(self.images_names)} images")
            if SSH:
                self.target_names = [x.replace('C:/Cours-Sorbonne/M1/Stage/src/','../').replace('similaires_rol_extracted_nn_compressed','sim_rol_super_compressed') for x in self.target_names.copy()]


    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        
        try:
            image_file = self.images_names[idx].split('.')[0]

            img = Image.open(os.path.join(self.path_rol,image_file)+".jpg").convert('L')
            target = None
            target_file = None
            
            if self.path_ht_rol is not None:
                target_image = Image.open(os.path.join(self.path_ht_rol,image_file)+".jpg").convert('L')
                target = self.transform_simclr(target_image)
            elif self.use_only_rol:
                target = self.transform_simclr(img)
            else:
                target_file = self.target_names[idx]
                target = Image.open(target_file.replace("\\","/")).convert('L')
                if self.augment_test:
                    target = self.transform_simclr(img)
                else:
                    target = self.transform(target)

            img = self.transform(img)

            if self.use_context:
                img_context, target_context = None,None
                img_context = JR.get_captions(image_file, self.path_rol)

                if not self.use_only_rol:
                    target_context = JR.get_captions(target_file, self.path_sim_rol_nn_extracted, target=True, augment=False)
                else:
                    target_context = JR.get_captions(image_file, self.path_rol, augment=True)

                if target_context is None or img_context is None:
                    print("[WARNING] No context provided")
                    target_context, img_context = "<UNK>"

                return img, target, img_context, target_context
            
            return img, target

        
        except Exception as e:
            print("[ERROR-GETITEM]", e)
            random_tensor = torch.ones((3,self.shape[0], self.shape[1]))
            return random_tensor,random_tensor


    def _get_test_files(self, path_to_sim_test):
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


    def _get_best_file(self, image_file, target_file) -> str:
        """
            Function to get the best target image for a given image using SIFT
            @param image_file
            @param target_file
            @return best_target_file
        """
        temp_path = os.path.join(self.path_sim_rol_nn_extracted,f'{target_file}*')
        target_files =  glob.glob(temp_path)
        
        if len(target_files) == 1:
            return target_files[0]
        elif len(target_files) > 1:
            best_file = None
            try:
                _, des = SIFTDetector.computeSIFT(cv2.imread(os.path.join(self.path_rol,image_file)+".jpg"))
                all_kp_des = [SIFTDetector.computeSIFT(cv2.imread(file)) for file in target_files]
                best_match = SIFTDetector.getBestMatch(des, [des[1] for des in all_kp_des])
                best_file = target_files[best_match[0][0]]
            except:
                best_file = target_files[0]
            finally:
                return best_file

    def _get_pairs(self, path):
        pairs = None
        with open(path,"r") as f :
            pairs = f.readlines()
        pairs = [x.replace('\n','') for x in pairs]
        return pairs
    
    def _remove_pairs(self, images, targets, pairs_to_remove):
        filtered_images = []
        filtered_targets = []
        for x, y in zip(images, targets):
            if x not in pairs_to_remove:
                filtered_images.append(x)
                filtered_targets.append(y)
        return filtered_images, filtered_targets

        
    def build_dataset(self, save_path="C:/Cours-Sorbonne/M1/Stage/src/data/rol_sim_rol_triplets/targets.npy"):
        """
            Function that builds the triplets (image, target)
            @param save_path : the path where to save the targets created
        """
        real_targets = []
        for image_file, target_file in tqdm(zip(self.images_names,self.target_names)):
            target_file = self._get_best_file(image_file, target_file)
            real_targets.append(target_file)
        self.target_names = real_targets.copy()
        np.save(save_path, real_targets)   


    @staticmethod
    def show_data(loader, nb_images=1, use_context=False, is_test=False):
        """
            Function that show data from a given loader
            @param loader
        """
        assert nb_images < 32, "[ERROR] nb_images should be less than 32"
        if use_context:
            if is_test:
                x, y,_,_,_,_ = next(iter(loader))
            else:
                x, y,_,_ = next(iter(loader))
        else:
            if is_test:
                x,y,_,_ = next(iter(loader))
            else:
                x,y = next(iter(loader))

        x = x.permute(0,2,3,1)
        y = y.permute(0,2,3,1)

        for i in range(nb_images):
            plt.figure(figsize=(12,7))
            plt.subplot(121)
            plt.imshow(x[i], cmap="gray")
            plt.title("Image")
            plt.subplot(122)
            plt.imshow(y[i], cmap="gray")
            plt.title("Target")
            plt.show()