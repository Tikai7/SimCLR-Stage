import cv2
import os
import concurrent.futures
import numpy as np 

class DataManager:
    """
    Class to manage the data of the dataset.
    """
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
    def read_and_compress_image(file, path, dest_path, shape):
        """
        Read an image from a file, resize it to the desired shape and save it to a destination path.
        The resizing is done by keeping the aspect ratio and filling the remaining space with black pixels.
        @param file: The name of the file to read.
        @param path: The path to the file.
        @param dest_path: The path to save the compressed image.
        @param shape: The desired shape of the image.
        """
        img = cv2.imread(os.path.join(path, file))
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1
        if h == w: 
            return cv2.resize(img, shape, cv2.INTER_AREA)
        
        dif = h if h > w else w
        interpolation = cv2.INTER_AREA if dif > (shape[0]+shape[1])//2 else  cv2.INTER_CUBIC
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2
        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        img = cv2.resize(mask, shape, interpolation)
        img_name = file.split('/')[-1]
        cv2.imwrite(os.path.join(dest_path, img_name), img)
    
    @staticmethod
    def read_and_compress_files(path, dest_path, shape=(1024, 1024), max_workers=4, single_folder=False):
        """
        Read all the images in a directory in parrallel, resize them to the desired shape and save them to a destination path.
        The resizing is done by keeping the aspect ratio and filling the remaining space with black pixels.
        @param path: The path to the directory containing the images.
        @param dest_path: The path to save the compressed images.
        @param shape: The desired shape of the images.
        @param max_workers: The maximum number of threads to use.
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
            rol_files = os.listdir(path)    
            rol_index = [f.split('rol')[1].zfill(2) for f in rol_files]
            all_files = [
                f"{path}/{file}/dir{index}/" + filename 
                for file, index in zip(rol_files, rol_index) 
                for filename in os.listdir(f'{path}/{file}/dir{index}')
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(DataManager.read_and_compress_image, file, path, dest_path, shape)
                    for file in all_files
                ]
                concurrent.futures.wait(futures)
