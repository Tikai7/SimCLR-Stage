import cv2
import numpy as np
import concurrent.futures
from PIL import Image

import scipy


class Processing:

    @staticmethod
    def median_smoothing(image):
        return cv2.medianBlur(image, 5)

    @staticmethod
    def fft_smoothing(image, radius=12):        
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        
        u, v = fft_image.shape
        center_u, center_v = u // 2, v // 2
        y, x = np.ogrid[:u, :v]
        mask = ((x - center_v)**2 + (y - center_u)**2) <= radius**2
        
        fft_shifted[~mask] = 0
        
        fft_unshifted = np.fft.ifftshift(fft_shifted)
        smoothed_image = np.real(np.fft.ifft2(fft_unshifted))
        
        smoothed_image -= smoothed_image.min()
        smoothed_image /= smoothed_image.max()
        
        smoothed_image = (smoothed_image * 255).astype(np.uint8)

        return smoothed_image

    @staticmethod
    def smooth_halftone_image(image, radius=12):
        image = np.array(image)
        if len(image.shape) == 3:
            channels = cv2.split(image)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(Processing.fft_smoothing, channel, radius) for channel in channels]
                smoothed_channels = [f.result() for f in concurrent.futures.as_completed(futures)]
            smoothed_channels = cv2.merge(smoothed_channels)
        else:
            smoothed_channels = Processing.fft_smoothing(image)
        
        return Image.fromarray(smoothed_channels)

    @staticmethod
    def to_halftone(image):
        image = np.array(image)
        if len(image.shape) == 3:
            channels = cv2.split(image)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(Processing.jarvis_judice_ninke_dithering, channel) for channel in channels]
                # for f in concurrent.futures.as_completed(futures):
                dithered_channels = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            dithered_image = cv2.merge(dithered_channels)
        else:
            dithered_image = Processing.jarvis_judice_ninke_dithering(image)
        
        return Image.fromarray(dithered_image)


    @staticmethod
    def jarvis_judice_ninke_dithering(image):
        # Define the error diffusion matrix for Jarvis, Judice, and Ninke dithering
        JJN_MATRIX = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]) / 48.0

        # Get image dimensions
        height, width = image.shape
        # Iterate through the image pixels
        for y in range(height):
            for x in range(width):
                old_pixel = image[y, x]
                new_pixel = 255 * (old_pixel > 127)
                image[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                for dy in range(3):
                    for dx in range(5):
                        ny, nx = y + dy, x + dx - 2
                        if 0 <= ny < height and 0 <= nx < width:
                            image[ny, nx] += quant_error * JJN_MATRIX[dy, dx]

        return image

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