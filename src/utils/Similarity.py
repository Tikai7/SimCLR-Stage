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


path_sift_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/rol" 
path_sift_sim_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/sim_rol" 
path_resnet_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/features_rol.npy"
path_resnet_sim_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/features_sim_rol.npy"
path_matches = "C:/Cours-Sorbonne/M1/Stage/src/params/matches.npy"

class Similarity:
    @staticmethod
    def match_images_with_mse(path, path_to_match):

        def compute_mse(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
            return err
    
        all_images = [cv2.imread(f"{path}/{img}") for img in os.listdir(path)]
        all_images_sim = [cv2.imread(f"{path_to_match}/{img}") for img in os.listdir(path_to_match)]
        for image in all_images:
            mse = np.inf
            for imgage_sim in all_images_sim:
                new_mse = compute_mse(image, imgage_sim)
                if new_mse < mse:
                    mse = new_mse
                    best_match = imgage_sim
            PL.plot_images([image, best_match], ["Original", "Best match"])

    @staticmethod
    def match_images_with_sift(path, path_to_match):
        try:
            all_images_rol_sift = SIFTDetector.loadSIFTDescriptors(path_sift_rol)
            all_images_sim_sift = SIFTDetector.loadSIFTDescriptors(path_sift_sim_rol)
        except:
            all_images_rol_sift = []
            all_images_sim_sift = []

        if not all_images_rol_sift: 
            print("Loading rol images...")
            all_images_rol = [cv2.imread(f"{path}/{img}") for img in os.listdir(path)[:1000]]
            print("Computing SIFT descriptors for all images in rol...")        
            all_images_rol_sift = [SIFTDetector.computeSIFT(img) for img in all_images_rol]
            all_images_rol_sift = [img for img in all_images_rol_sift if img[1] is not None]
            SIFTDetector.saveSIFTDescriptors(all_images_rol_sift, path_sift_rol)

        if not all_images_sim_sift:
            print("Loading sim images...")
            all_images_sim = [cv2.imread(f"{path_to_match}/{img}") for img in os.listdir(path_to_match)]
            print("Computing SIFT descriptors for all images in sim rol...")
            all_images_sim_sift = [SIFTDetector.computeSIFT(img) for img in all_images_sim]
            all_images_sim_sift = [img for img in all_images_sim_sift if img[1] is not None]
            SIFTDetector.saveSIFTDescriptors(all_images_sim_sift, path_sift_sim_rol)

        for image, image_sift in tqdm(zip(all_images_rol,all_images_rol_sift)):
            best_match = SIFTDetector.getBestMatch(
                image_sift[1],
                [img[1] for img in all_images_sim_sift]
            )
            SIFTDetector.displayKeypoints(
                image,
                image_sift[0],
                all_images_sim[best_match[0]],
                all_images_sim_sift[best_match[0]][0],
                best_match[2]
            )
            

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