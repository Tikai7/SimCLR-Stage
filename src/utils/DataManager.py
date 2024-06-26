
import cv2
import os
import concurrent.futures
from utils.Plotter import Plotter as PL
from utils.Processing import Processing as PR
from utils.ImageExtractor import ImageExtractor as IE
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
        all_images = []
        for file in all_files:
            image = cv2.imread(os.path.join(path, file))
            overlay_image, mask, extracted_images = IE.extract(image)
            PL.plot_images([image, mask, overlay_image, extracted_images], ['Image', 'Mask', 'Overlay',' Extracted Images'])    
            # all_images.extend(images)
            # break

        