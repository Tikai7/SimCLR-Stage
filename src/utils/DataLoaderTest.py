import os
import torch
import numpy as np 
from torch.utils.data import Dataset
from torchvision import transforms
from utils.Processing import Processing as PC
from utils.JSONRetriever import JSONRetriever as JR
from PIL import Image

class DataLoaderTest(Dataset):

    def __init__(self, 
            path_rol="C:/Cours-Sorbonne/M1/Stage/src/data/rol_compressed" , 
            path_sim_rol= "C:/Cours-Sorbonne/M1/Stage/src/data/similaires_rol_extracted_nn_compressed",
            path_to_sim_test="",shape=(256,256), augment=False
        ) -> None:
        super().__init__()
        self.all_files = os.listdir(path_to_sim_test)
        self.all_files = sorted(self.all_files, key=lambda x:int(x.split("ID_")[1].split('.')[0]))
        self.shape = shape 
        self.augment = augment
        self.path_sim_test = path_to_sim_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_rol = path_rol
        self.path_sim_rol_nn_extracted = path_sim_rol

        self.transform = transforms.Compose([
                transforms.Resize(self.shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ]) 

        self.augment_transform = transforms.Compose([
                transforms.Resize(self.shape),
                transforms.Lambda(lambda x : PC.apply_rotogravure_effect(x, method="grid", intensity=128, dot_size=2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ]) 

        
        self.target_names = []
        self.image_names = []
        for i in range(0,len(self.all_files)-1,2):
            target = self.all_files[i]
            img = self.all_files[i+1]

            if not img.startswith("btv"):
                img, target = target, img

            self.target_names.append(target)
            self.image_names.append(img)

        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_file = self.image_names[index]
        target_file = self.target_names[index]

        img = Image.open(f"{self.path_sim_test}/{img_file}").convert("L")
        target = Image.open(f"{self.path_sim_test}/{target_file}").convert("L")

        img = transforms.Resize(self.shape)(img)
        target = transforms.Resize(self.shape)(target)

        img_t = self.transform(img)
        target_t = self.transform(target) if not self.augment else self.augment_transform(target)
        
        img_file = img_file.split("_ID")[0]
        target_file = target_file.split("_ID")[0].replace(".jpg","")
        img_context = JR.get_captions(img_file, self.path_rol)
        target_context = JR.get_captions(target_file, self.path_sim_rol_nn_extracted, augment=False)

        return img_t, target_t, np.array(img), np.array(target), img_context, target_context
