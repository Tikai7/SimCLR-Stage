import os
import cv2
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import BertTokenizer

from model.SIFT import SIFTDetector
from tqdm import tqdm
from utils.Plotter import Plotter as PL 
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from model.BERT import BertEncoder


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

path_sift_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/rol" 
path_sift_sim_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/sim_rol" 
path_resnet_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/features_rol.npy"
path_resnet_sim_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/features_sim_rol.npy"

path_matches = "C:/Cours-Sorbonne/M1/Stage/src/params/matches.npy"
path_matches_mse = "C:/Cours-Sorbonne/M1/Stage/src/params/matches_mse.npy"
path_matches_sift = "C:/Cours-Sorbonne/M1/Stage/src/params/matches_sift.npy"

class Similarity:
    """
    Similarity class to compute similarity between images
    
    Methods
    -------
    match_images_with_mse(path, path_to_match, plot=False, nb_plot=10)
        Match images using Mean Squared Error
    match_images_with_sift(path, path_to_match, plot=False, nb_plot=10, only_features=False)
        Match images using SIFT
    match_images_with_resnet(path, path_to_match, threshold=1000, max_images=-1, plot=False, nb_plot=10, only_features=False)
        Match images using ResNet50
    match_images_with_simCLR(model, test_loader=None, alpha=0.5, k=10, use_sift=False, is_test=False)
        Match images using SimCLR
    """


    @staticmethod
    def match_images_with_mse(path, path_to_match, plot=False, nb_plot=10):
        """
        Match images using Mean Squared Error
        Args:
        -----
            path (str): Path to the dataset
            path_to_match (str): Path to the dataset to match
            plot (bool): Whether to plot the images or not
            nb_plot (int): Number of images to plot
        Returns:
        --------
            best_matches (list): List of best matches
        """

        def compute_mse(imageA, imageB):
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
            return err
        
        try:
            best_matches = np.load(path_matches_mse)
            return best_matches
        except:
            pass
        
        all_images = [cv2.imread(f"{path}/{img}") for img in os.listdir(path)]
        all_images_sim = [cv2.imread(f"{path_to_match}/{img}") for img in os.listdir(path_to_match)]
        best_matches = []
        for image in all_images:
            mse = np.inf
            for imgage_sim in all_images_sim:
                new_mse = compute_mse(image, imgage_sim)
                if new_mse < mse:
                    mse = new_mse
                    best_match = imgage_sim
            best_matches.append(best_match)

        if plot : 
            for i, image, best_match in enumerate(zip(all_images, best_matches)):
                PL.plot_images([image, best_match], ["Original", "Best match"])
                if i > nb_plot:
                    break
        
        np.save(path_matches_mse, np.array(best_matches))
        return best_matches

    @staticmethod
    def match_images_with_sift(path, path_to_match, plot=False, nb_plot=10, only_features=False):

        """
        Match images using SIFT
        Args:
        -----
            path (str): Path to the dataset
            path_to_match (str): Path to the dataset to match
            plot (bool): Whether to plot the images or not
            nb_plot (int): Number of images to plot
            only_features (bool): Whether to return only the features or not
        Returns:
        --------
            best_matches (list): List of best matches
        """

        try:
            best_matches = np.load(path_matches_sift)
            return best_matches
        except:
            pass

        try:
            all_images_rol_sift = SIFTDetector.loadSIFTDescriptors(path_sift_rol)
            all_images_sim_sift = SIFTDetector.loadSIFTDescriptors(path_sift_sim_rol)
            if only_features:
                return all_images_rol_sift, all_images_sim_sift
        except:
            all_images_rol_sift = []
            all_images_sim_sift = []

        if not all_images_rol_sift: 
            print("Loading rol images...")
            all_images_rol = [cv2.imread(f"{path}/{img}") for img in os.listdir(path)]
            print("Computing SIFT descriptors for all images in rol...")        
            all_images_rol_sift = [SIFTDetector.computeSIFT(img) for img in all_images_rol]
            all_images_rol_sift = [img for img in all_images_rol_sift if img[1] is not None]
            # SIFTDetector.saveSIFTDescriptors(all_images_rol_sift, path_sift_rol)

        if not all_images_sim_sift:
            print("Loading sim images...")
            all_images_sim = [cv2.imread(f"{path_to_match}/{img}") for img in os.listdir(path_to_match)]
            print("Computing SIFT descriptors for all images in sim rol...")
            all_images_sim_sift = [SIFTDetector.computeSIFT(img) for img in all_images_sim]
            all_images_sim_sift = [img for img in all_images_sim_sift if img[1] is not None]
            SIFTDetector.saveSIFTDescriptors(all_images_sim_sift, path_sift_sim_rol)

        if only_features:
            return all_images_rol_sift, all_images_sim_sift
        
        if plot :
            i = 0
            best_matches = []
            for image, image_sift in tqdm(zip(all_images_rol,all_images_rol_sift)):
                best_match = SIFTDetector.getBestMatch(
                    image_sift[1],
                    [img[1] for img in all_images_sim_sift]
                )
                best_matches.append(best_match)
                SIFTDetector.displayKeypoints(
                    image,
                    image_sift[0],
                    all_images_sim[best_match[0]],
                    all_images_sim_sift[best_match[0]][0],
                    best_match[2]
                )
                if i > nb_plot:
                    break
                i += 1
            
        return best_matches

    @staticmethod
    def match_images_with_resnet(path, path_to_match, threshold=1000, max_images=-1, plot=False, nb_plot=10, only_features=False):

        """
        Match images using ResNet50
        Args:
        -----
            path (str): Path to the dataset
            path_to_match (str): Path to the dataset to match
            threshold (int): Threshold for the distance
            max_images (int): Maximum number of images to consider
            plot (bool): Whether to plot the images or not
            nb_plot (int): Number of images to plot
            only_features (bool): Whether to return only the features or not
        Returns:
        --------
            matches (list): List of best matches
        """

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
            
        def extract_features_for_dataset(dataset_path, path_to_load, max_images=None):
            features = []
            image_paths = []
            need_to_compute_features = True
            try:
                features = np.load(path_to_load)
                need_to_compute_features = False
            except:
                pass

            all_images = os.listdir(dataset_path) if max_images is None else os.listdir(dataset_path)[:max_images]
            for img_name in all_images:
                img_path = os.path.join(dataset_path, img_name)
                f = None
                if need_to_compute_features:
                    f = extract_features(img_path, model)
                    if f is not None: 
                        features.append(f)
                # if f is not None:
                if (need_to_compute_features and f is not None) or not need_to_compute_features:
                    image_paths.append(img_path)
                    
            if need_to_compute_features:
                features = np.array(features)
            return features, image_paths

        print("Extracting features for sim images...")
        features_small, image_paths_small = extract_features_for_dataset(path_to_match, path_resnet_sim_rol, None)
        np.save(path_resnet_sim_rol, features_small)

        print("Extracting features for rol images...")
        features_big, image_paths_big = extract_features_for_dataset(path, path_resnet_rol, max_images)
        np.save(path_resnet_rol, features_big)

        if only_features:
            return features_big, features_small, image_paths_big, image_paths_small
        
        print("Computing nearest neighbors...")
        
        try:
            matches = np.load(path_matches)
            if matches.shape[0] < 100:
                raise Exception
        except:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(features_small)
            distances, indices = nbrs.kneighbors(features_big)
            matches = [(image_paths_big[i], image_paths_small[indices[i][0]]) for i in range(len(image_paths_big)) if distances[i][0] < threshold]
            np.save(path_matches, matches)

        if plot : 
            for i,match in enumerate(matches):
                PL.plot_images([cv2.imread(match[0]), cv2.imread(match[1])], ["Original", "Best match"])
                if i > nb_plot:
                    break
        return matches, features_big, features_small
    
    
    @staticmethod
    def get_distance_between_images(image1, image2):
        return np.linalg.norm(image1 - image2)


    @staticmethod
    def find_most_similar(sim_matrix):
        most_similar_pairs = []
        num_originals = sim_matrix.size(0) // 2
        for i in range(num_originals):
            sim_scores = sim_matrix[i, num_originals:] 
            most_similar_idx = torch.argmax(sim_scores).item() + num_originals
            most_similar_pairs.append((i, most_similar_idx))
        return most_similar_pairs


    @staticmethod
    def match_images_with_simCLR(model, test_loader=None, alpha=0.5, k=10, use_sift=False, is_test=False):
        """
        Match images using SimCLR
        Args:
        -----
            model (torch.nn.Module): SimCLR model
            test_loader (torch.utils.data.DataLoader): Test loader
            path (str): Path to the dataset
            path_to_match (str): Path to the dataset to match
            use_context (bool): Whether to use context or not
            alpha (float): Alpha value
            k (int): Number of top-k matches
            use_sift (bool): Whether to use SIFT or not
            is_test (bool): Whether that we are using a real test loader or not
        Returns:
        --------
            top_k_indices (torch.Tensor): Top-k indices
            original_images (torch.Tensor): Original images
            augmented_images (torch.Tensor): Augmented images
            precisions (list): List of precisions
        """
       
       
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Matching on {device}")
        model.to(device)
        model.eval()

        bert = BertEncoder()
        bert.to(device)
        bert.eval()

        with torch.no_grad():
            original_text_features, augmented_text_features = [], []
            original_features, augmented_features = [], []
            original_images, augmented_images = [], []
            original_images_sift, augmented_images_sift = [], []

            for data in tqdm(test_loader):
                output = None

                if is_test:
                    batch_x, batch_y, batch_w, batch_z, context_x, context_y = data
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    batch_w, batch_z = batch_w.to(device), batch_z.to(device)
                    context_x = tokenizer(list(context_x), padding=True, return_tensors='pt', add_special_tokens=True)
                    context_y = tokenizer(list(context_y), padding=True, return_tensors='pt', add_special_tokens=True)

                    output = model(batch_x, batch_y, context_x, context_y)
                else:
                    batch_x, batch_y, context_x, context_y = data
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    context_x = tokenizer(list(context_x), padding=True, return_tensors='pt', add_special_tokens=True)
                    context_y = tokenizer(list(context_y), padding=True, return_tensors='pt', add_special_tokens=True)

                    output = model(batch_x, batch_y, context_x, context_y)

                Z1, Z2 = output['projection_head']



                original_features.append(Z1.cpu())
                augmented_features.append(Z2.cpu())
                original_images.append(batch_x.cpu())
                augmented_images.append(batch_y.cpu())
                original_text_features.append(bert(context_x).cpu())
                augmented_text_features.append(bert(context_y).cpu())
                
                if use_sift:
                    original_images_sift.append(batch_w.cpu())
                    augmented_images_sift.append(batch_z.cpu())


        original_features = torch.cat(original_features, dim=0)
        augmented_features = torch.cat(augmented_features, dim=0)
        original_images = torch.cat(original_images, dim=0)
        augmented_images = torch.cat(augmented_images, dim=0)
        original_text_features = torch.cat(original_text_features, dim=0)
        augmented_text_features = torch.cat(augmented_text_features, dim=0)

        if use_sift:
            original_images_sift = torch.cat(original_images_sift, dim=0)
            augmented_images_sift = torch.cat(augmented_images_sift, dim=0)
            original_sift_features = [SIFTDetector.computeSIFT(img.numpy()) for img in original_images_sift]
            augmented_sift_features = [SIFTDetector.computeSIFT(img.numpy()) for img in augmented_images_sift]


        sim_matrix_text = F.cosine_similarity(original_text_features.unsqueeze(1), augmented_text_features.unsqueeze(0), dim=-1)
        sim_matrix = F.cosine_similarity(original_features.unsqueeze(1), augmented_features.unsqueeze(0), dim=-1)

        sim_matrix_combined = alpha*sim_matrix + (1-alpha)*sim_matrix_text
        top_k_indices = torch.topk(sim_matrix_combined, k, dim=1).indices
        true_indices = torch.arange(len(top_k_indices)).unsqueeze(1)

        if use_sift :
            for idx, (_, des) in enumerate(original_sift_features):
                if des is not None:
                    best_matches = SIFTDetector.getBestMatch(des, [augmented_sift_features[i][1] for i in top_k_indices[idx]])
                    top_k_indices[idx] = torch.tensor([top_k_indices[idx][match[0]] for match in best_matches])

        precisions = []
        for i in range(0, k + 1, 5):
            top_n = i if i != 0 else 1
            current_top_k_indices = top_k_indices[:, :top_n]
            top_n_correct = (current_top_k_indices == true_indices).sum().item()
            precision_top_n = top_n_correct / len(current_top_k_indices)
            precisions.append(precision_top_n)
            print(f"[INFO] Top-{top_n} Precision: {precision_top_n}")

        return top_k_indices, original_images, augmented_images, precisions


        
