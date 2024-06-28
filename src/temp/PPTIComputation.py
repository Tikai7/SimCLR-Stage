import os
import cv2
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.SIFT import SIFTDetector
from tqdm import tqdm
from utils.Plotter import Plotter as PL 
from PIL import Image
from sklearn.neighbors import NearestNeighbors


path_rol_comp = "/tempory/M1-DAC-Stage-Tikai7/Stage-M1/src/data/rol_compressed"
path_sim_rol_extract_comp = "/tempory/M1-DAC-Stage-Tikai7/Stage-M1/src/data/similaires_rol_extracted_compressed"

path_resnet_rol = "/tempory/M1-DAC-Stage-Tikai7/Stage-M1/src/params/features_rol.npy"
path_resnet_sim_rol = "/tempory/M1-DAC-Stage-Tikai7/Stage-M1/src/params/features_sim_rol.npy"
path_matches = "/tempory/M1-DAC-Stage-Tikai7/Stage-M1/src/params/matches.npy"

class PPTI:
    @staticmethod
    def match_images_with_nn(path, path_to_match, threshold=1000, max_images=-1, plot=False):
        model = models.resnet50(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def extract_features(img_path, model):
            try : 
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0)
                with torch.no_grad():
                    batch_t = batch_t.to(device)
                    features = model(batch_t)
                return features.cpu().numpy().flatten()
            except:
                return None
            
        def extract_features_for_dataset(dataset_path, path_to_load, max_images=-1):
            features = []
            image_paths = []
            need_to_compute_features = True
            try:
                features = np.load(path_to_load)
                need_to_compute_features = False
            except:
                pass
            for img_name in os.listdir(dataset_path)[:max_images]:
                img_path = os.path.join(dataset_path, img_name)
                if need_to_compute_features:
                    f = extract_features(img_path, model)
                    if f is not None: 
                        features.append(f)
                if f is not None:
                    image_paths.append(img_path)
            if need_to_compute_features:
                features = np.array(features)
            return features, image_paths

        print("Extracting features for sim images...")
        features_small, image_paths_small = extract_features_for_dataset(path_to_match, path_resnet_sim_rol, max_images)
        np.save(path_resnet_sim_rol, features_small)

        print("Extracting features for rol images...")
        features_big, image_paths_big = extract_features_for_dataset(path, path_resnet_rol, max_images)
        np.save(path_resnet_rol, features_big)

        print("Computing nearest neighbors...")
        
        try:
            matches = np.load(path_matches)
        except:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(features_small)
            distances, indices = nbrs.kneighbors(features_big)
            matches = [(image_paths_big[i], image_paths_small[indices[i][0]]) for i in range(len(image_paths_big)) if distances[i][0] < threshold]
            np.save(path_matches, matches)

        if plot : 
            for match in matches:
                PL.plot_images([cv2.imread(match[0]), cv2.imread(match[1])], ["Original", "Best match"])
                
        return matches, features_big, features_small



PPTI.match_images_with_nn(path_rol_comp, path_sim_rol_extract_comp, max_images=-1, plot=True)