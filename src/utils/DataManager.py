from tqdm import tqdm
import cv2
import os
import numpy as np
import concurrent.futures
import glob
import matplotlib.pyplot as plt
from utils.ImageExtractor import ImageExtractor as IE
from utils.JSONRetriever import JSONRetriever as JR
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model.SIFT import SIFTDetector

class DataManager(Dataset):
    """
    Class to manage the data of the dataset.
    """
    def __init__(
            self, path_rol, path_sim_rol_nn_extracted, path_json_filtered, 
            shape=(256,256), target_path="C:/Cours-Sorbonne/M1/Stage/src/data/rol_sim_rol_triplets/targets.npy"
    ) -> None:
    
        self.path_filtered = path_json_filtered
        self.path_rol = path_rol
        self.path_sim_rol_nn_extracted = path_sim_rol_nn_extracted
        self.shape = shape

        try:
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
            np.save(images_path, self.images_names)
            self.build_dataset(target_path)        
            
        # self.images_names = self.images_names[:len(self.target_names)]

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        
        try:
            image_file = self.images_names[idx].split('.')[0]
            target_file = self.target_names[idx]

            img = Image.open(os.path.join(self.path_rol,image_file)+".jpg").convert('RGB')
            target = Image.open(target_file).convert('RGB')

            img = self.transform(img)
            target = self.transform(target)
            return img, target
        except Exception as e:
            print(e)
            return None, None


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


    def transform(self, image):
        """
            Function that apply transformation on an given Image
            @param Image
            @return Augmented Image
        """
        f =  transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(     
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
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
    def show_data(loader):
        """
            Function that show data from a given loader
            @param loader
        """
        x, y = next(iter(loader))

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
        
    @staticmethod
    def get_missing_files(path,path_comp):
        """
        Get the files that are in the original dataset but are not in the compressed dataset.
        @param path: The path to the original dataset.
        @param path_comp: The path to the compressed dataset.
        """
        rol_files = os.listdir(path)
        rol_index = [f.split('rol')[1].zfill(2) for f in rol_files]
        all_files = [
            filename 
            for file, index in zip(rol_files, rol_index) 
            for filename in os.listdir(f'{path}/{file}/dir{index}')
        ]
        return (set(all_files).difference(set(os.listdir(path_comp))))


    @staticmethod
    def read_image(file, path):
        """ Read an image from a specified path. """
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"No image found at {img_path}")
        return img

    @staticmethod
    def resize_image(img, shape):
        """ Resize image to a specific shape, maintaining aspect ratio. """
        h, w = img.shape[:2]
        aspect_ratio = h / w
        target_h, target_w = shape
        if aspect_ratio > 1:
            new_h, new_w = target_h, int(target_h / aspect_ratio) 
        else:
            new_w, new_h = target_w, int(target_w * aspect_ratio)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_img

    @staticmethod
    def add_borders(img, shape):
        """ Add black borders to maintain the desired aspect ratio. """
        h, w = img.shape[:2]
        top = bottom = (shape[0] - h) // 2
        left = right = (shape[1] - w) // 2
        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_img

    @staticmethod
    def save_image(img, dest_path, file_name):
        """ Save the image to the destination path. """
        img_name = file_name.split('/')[-1]
        full_path = os.path.join(dest_path, img_name)
        cv2.imwrite(full_path, img)

    @staticmethod
    def read_and_compress_image(file, path, dest_path, shape):
        """
        Read an image from a file, resize it to the desired shape and save it to a destination path.
        The resizing is done by keeping the aspect ratio and filling the remaining space with black pixels.
        @param file: The name of the file to read.
        @param path: The path to the file.
        @param dest_path: The path to save the compressed image.
        @param shape: The desired shape of the image.
        """
        img = DataManager.read_image(file, path)
        resized_img = DataManager.resize_image(img, shape)
        final_img = DataManager.add_borders(resized_img, shape)
        DataManager.save_image(final_img, dest_path, file)
    
    @staticmethod
    def read_and_compress_files(path, dest_path, shape=(1024, 1024), max_workers=4, single_folder=False):
        """
        Read all the images in a directory in parrallel, resize them to the desired shape and save them to a destination path.
        The resizing is done by keeping the aspect ratio and filling the remaining space with black pixels.
        @param path: The path to the directory containing the images.
        @param dest_path: The path to save the compressed images.
        @param shape: The desired shape of the images.
        @param max_workers: The maximum number of threads to use.
        @param single_folder: If the images are in a single folder or in multiple folders.
        """
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        if single_folder:
            all_files = os.listdir(path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(DataManager.read_and_compress_image, file, path, dest_path, shape)
                    for file in all_files
                ]
                concurrent.futures.wait(futures)
        else:
            computed_files = set(os.listdir(dest_path))
            rol_files = os.listdir(path)    
            rol_index = [f.split('rol')[1].zfill(2) for f in rol_files]
            all_files = [
                f"{path}/{file}/dir{index}/" + filename 
                for file, index in zip(rol_files, rol_index) 
                for filename in os.listdir(f'{path}/{file}/dir{index}')
                if filename not in computed_files
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(DataManager.read_and_compress_image, file, path, dest_path, shape)
                    for file in all_files
                ]
                concurrent.futures.wait(futures)

    @staticmethod
    def extract_images(path, dest_path):
        """
        Extract images from a compressed file to a destination path.
        @param path: The path to the compressed file.
        @param dest_path: The path to save the extracted images.
        """
        all_files = os.listdir(path)
        all_processed_file = set(os.listdir(dest_path))
        all_processed_file = set([file.split('_')[0] for file in all_processed_file])
        for file in tqdm(all_files):
            if file.split('_')[0] in all_processed_file:
                continue
            try : 
                image = cv2.imread(os.path.join(path, file))
                extracted_images = IE.extract_easy(image)
                for i, img in enumerate(extracted_images):
                    DataManager.save_image(img, dest_path, f"{file.split('.')[0]}_{i}.png")
            except:
                continue
            