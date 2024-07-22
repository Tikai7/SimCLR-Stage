import cv2
import numpy as np
import concurrent.futures
from PIL import Image
import random



class RandomRotation90:
    def __init__(self):
        pass
    
    def __call__(self, img):
        # Randomly choose between 90 or -90 degrees
        angle = random.choice([90, -90])
        return img.rotate(angle, expand=True)
    
class Processing:

    @staticmethod
    def apply_rotogravure_effect(image, dot_size=2, intensity=128):
        image = np.array(image)
        normalized_image = image / 255.0

        dot_pattern = np.zeros((dot_size, dot_size), dtype=np.float32)
        cv2.circle(dot_pattern, (dot_size // 2, dot_size // 2), dot_size // 4, 1, -1)

        tiled_pattern = np.tile(dot_pattern, (image.shape[0] // dot_size + 1, image.shape[1] // dot_size + 1))
        tiled_pattern = tiled_pattern[:image.shape[0], :image.shape[1]]

        pattern_intensity = intensity / 255.0
        rotogravure_effect = normalized_image * (1 - pattern_intensity) + tiled_pattern * pattern_intensity

        rotogravure_effect = (rotogravure_effect * 255).astype(np.uint8)

        return Image.fromarray(rotogravure_effect)

    @staticmethod
    def to_halftone(image):
        image = np.array(image)
        if len(image.shape) == 3:
            channels = cv2.split(image)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(Processing.floyd_steinberg_dithering, channel) for channel in channels]
                # for f in concurrent.futures.as_completed(futures):
                dithered_channels = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            dithered_image = cv2.merge(dithered_channels)
        else:
            dithered_image = Processing.floyd_steinberg_dithering(image)
        
        return Image.fromarray(dithered_image)

    @staticmethod
    def floyd_steinberg_dithering(channel):
        
        channel = channel / 255.0
        height, width = channel.shape

        for y in range(height):
            for x in range(width):
                old_pixel = channel[y, x]
                new_pixel = np.round(old_pixel)
                channel[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                if x + 1 < width:
                    channel[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height and x - 1 >= 0:
                    channel[y + 1, x - 1] += quant_error * 3 / 16
                if y + 1 < height:
                    channel[y + 1, x] += quant_error * 5 / 16
                if y + 1 < height and x + 1 < width:
                    channel[y + 1, x + 1] += quant_error * 1 / 16

        channel = (channel * 255).astype(np.uint8)
        return channel


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