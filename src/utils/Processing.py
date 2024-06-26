import cv2
import numpy as np

class Processing:
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