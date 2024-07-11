from tqdm import tqdm
import cv2
import os
import numpy as np
import glob
import torch
from utils.JSONRetriever import JSONRetriever as JR
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model.SIFT import SIFTDetector
import matplotlib.pyplot as plt

SSH = os.getcwd() != 'c:\\Cours-Sorbonne\\M1\\Stage\\src'

class DataLoaderSimCLR(Dataset):
    def __init__(
            self, path_rol, path_sim_rol_nn_extracted, path_json_filtered, 
            shape=(256,256), target_path="C:/Cours-Sorbonne/M1/Stage/src/data/rol_sim_rol_triplets/targets.npy", 
            sim_clr=False, use_only_rol=False, build_if_error = False, max_images=None
    ) -> None:

        self.use_only_rol = use_only_rol
        self.sim_clr = sim_clr
        self.path_filtered = path_json_filtered
        self.path_rol = path_rol
        self.path_sim_rol_nn_extracted = path_sim_rol_nn_extracted
        self.shape = shape

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
            for x,y in zip(self.images_names.copy(), self.target_names.copy()):
                if x is None or y is None :
                    self.images_names.remove(x)
                    self.target_names.remove(y)
            if SSH:
                self.target_names = [x.replace('C:/Cours-Sorbonne/M1/Stage/src/','../').replace('similaires_rol_extracted_nn_compressed','sim_rol_super_compressed') for x in self.target_names.copy()]


    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        
        try:
            image_file = self.images_names[idx].split('.')[0]
            img = Image.open(os.path.join(self.path_rol,image_file)+".jpg").convert('RGB')

            target = None
            if self.use_only_rol:
                target = self.transform(img, sim_clr=self.sim_clr)
            else:
                target_file = self.target_names[idx]
                target = Image.open(target_file.replace("\\","/")).convert('RGB')
                target = self.transform(target, sim_clr=self.sim_clr)

            img = self.transform(img)

            return img, target
        except Exception as e:
            print("[ERROR-GETITEM]", e)
            random_tensor = torch.ones((3,self.shape[0], self.shape[1]))
            return random_tensor, random_tensor

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



    def transform(self, image, sim_clr=False):
        """
            Function that apply transformation on an given Image
            @param Image
            @return Augmented Image
        """
        f = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if sim_clr:
            f = transforms.Compose([
                transforms.RandomResizedCrop(size=self.shape, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * self.shape[0]), sigma=(0.1, 2.0)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        return f(image)
    
        
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
    def show_data(loader, nb_images=1):
        """
            Function that show data from a given loader
            @param loader
        """

        for i, (x, y) in enumerate(loader):
            if i == nb_images:
                break
            
            x = x.permute(0,2,3,1)
            y = y.permute(0,2,3,1)

            plt.figure(figsize=(12,7))
            plt.subplot(121)
            plt.imshow(x[0])
            plt.title("Image")
            plt.subplot(122)
            plt.imshow(y[0])
            plt.title("Target")
            plt.show()