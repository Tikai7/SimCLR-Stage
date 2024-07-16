import cv2
import numpy as np
from PIL import Image
import multiprocessing
from functools import partial

class Processing:

    @staticmethod
    def _diffusion_process(channel_data, height, width):
        diffusion_matrix = np.array([
            [0, 1, 7 / 16],
            [-1, 1, 3 / 16],
            [0, 1, 5 / 16],
            [1, 1, 1 / 16]
        ])
        
        for y in range(height):
            for x in range(width):
                old_pixel = channel_data[y, x]
                new_pixel = 255 if old_pixel > 127 else 0
                channel_data[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                for dx, dy, factor in diffusion_matrix:
                    nx, ny = int(x + dx), int(y + dy)
                    if 0 <= nx < width and 0 <= ny < height:
                        channel_data[ny, nx] = np.clip(channel_data[ny, nx] + quant_error * factor, 0, 255)
        
        return channel_data

    @staticmethod
    def to_halftone(image, max_workers=None):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        images_np = np.array(image)
        height, width, channels = images_np.shape
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            partial_func = partial(Processing._diffusion_process, height=height, width=width)
            results = pool.map(partial_func, [images_np[:, :, channel] for channel in range(channels)])
        
        stacked_image = np.stack(results, axis=-1)
        return Image.fromarray(stacked_image.astype('uint8'))
    
    ## ----------------- Oscar's PRAT project, Image Processing Functions ----------------- ##
    @staticmethod
    def postprocessing_mask(predicted_mask_resized):
        # Deriving binary masks for each class from the segmentation map
        background_mask = np.where(predicted_mask_resized == 0, 1, 0).astype(np.uint8)
        text_mask = np.where(predicted_mask_resized == 1, 1, 0).astype(np.uint8)
        image_mask = np.where(predicted_mask_resized == 2, 1, 0).astype(np.uint8)

        # Defining kernels for opening and closing
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))

        # Processing binary mask for each class
        refined_background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel1)
        refined_background_mask = cv2.morphologyEx(refined_background_mask, cv2.MORPH_CLOSE, kernel1)

        refined_image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel1)
        refined_image_mask = cv2.morphologyEx(refined_image_mask, cv2.MORPH_CLOSE, kernel2)

        refined_text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel3)
        refined_text_mask = cv2.morphologyEx(refined_text_mask, cv2.MORPH_CLOSE, kernel4)

        # Reconstructing the segmentation map from the refined binary masks and creating new overlay
        refined_mask = np.zeros_like(predicted_mask_resized)
        refined_mask[refined_background_mask == 1] = 0
        refined_mask[refined_text_mask == 1] = 1
        refined_mask[refined_image_mask == 1] = 2

        return refined_mask