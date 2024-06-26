import cv2
import os
from model.SIFT import SIFTDetector
from tqdm import tqdm


path_sift_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/rol"
path_sift_sim_rol = "C:/Cours-Sorbonne/M1/Stage/src/params/sim_rol"

class Similarity:
    @staticmethod
    def match_images(path, path_to_match):

        all_images_rol_sift = SIFTDetector.loadSIFTDescriptors(path_sift_rol)
        all_images_sim_sift = SIFTDetector.loadSIFTDescriptors(path_sift_sim_rol)

        if not all_images_rol_sift: 
            print("Loading rol images...")
            all_images_rol = [cv2.imread(f"{path}/{img}") for img in os.listdir(path)[:1000]]
            print("Computing SIFT descriptors for all images in rol...")        
            all_images_rol_sift = [SIFTDetector.computeSIFT(img) for img in all_images_rol]
            SIFTDetector.saveSIFTDescriptors(all_images_sim_sift, path_sift_sim_rol)

        if not all_images_sim_sift:
            print("Loading sim images...")
            all_images_sim = [cv2.imread(f"{path_to_match}/{img}") for img in os.listdir(path_to_match)]
            print("Computing SIFT descriptors for all images in sim rol...")
            all_images_sim_sift = [SIFTDetector.computeSIFT(img) for img in all_images_sim]
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
            