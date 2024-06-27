import tensorflow as tf
import numpy as np
import cv2
from utils.Processing import Processing as PR

# Loading the model of Oscar's PRAT project
model_path = "C:\Cours-Sorbonne\M1\Stage\doc\old_project_info\Archive\project-source-code-oskar-schade-zip_2024-02-19_1354\Project data\Final_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

class ImageExtractor:

    @staticmethod
    def extract_easy(image):
        """
        Extracts the images from a given image.
        (Uses a simple method to extract the images.)
        @param image: The image to extract the images from.
        @return: The extracted images.
        """
        all_images = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 700000:  # value to adjust as needed
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 1.5:  # Typical aspect ratio for images (GPT)
                    cropped_image = image[y:y+h, x:x+w]
                    all_images.append(cropped_image)
        return all_images


    ## ----------------- Oscar's PRAT project, Image Processing Functions ----------------- ##

    @staticmethod
    def extract(image):
        """
        Extracts the images from a given image.
        (Uses the model to predict the mask and then extracts the images.)
        @param image: The image to extract the images from.
        @return: The extracted images.
        """

        image_preprocessed, padding_info = ImageExtractor.pad_to_divisible(image)
        prepared_image = np.expand_dims(image_preprocessed, axis=0)
        predicted_mask = model.predict(prepared_image)
        predicted_mask = np.argmax(predicted_mask, axis=-1)[0]

        if padding_info[0] > 0 or padding_info[1] > 0:
            height, width = predicted_mask.shape
            predicted_mask = predicted_mask[:height - padding_info[0], :width - padding_info[1]]

        predicted_mask_resized = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        refined_mask = PR.postprocessing_mask(predicted_mask_resized)
        overlay_image = ImageExtractor.overlay_result(image, refined_mask)

        # Extracting image regions
        image_mask = (refined_mask == 2).astype(np.uint8) * 255
        extracted_images = cv2.bitwise_and(image, image, mask=image_mask)

        return overlay_image, refined_mask, extracted_images
    
    @staticmethod
    # Defining function to pad image to nearest suitable dimensions (divisible by 2n)
    def pad_to_divisible(image, n=5):
        new_height = ((image.shape[0] - 1) // (2**n) + 1) * (2**n)
        new_width = ((image.shape[1] - 1) // (2**n) + 1) * (2**n)

        pad_height = new_height - image.shape[0]
        pad_width = new_width - image.shape[1]

        padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Using zero-padding
        return padded_image, (pad_height, pad_width)

    @staticmethod
    # Defining function to overlay the predicted mask on the image, with colors indicating classes
    def overlay_result(image, mask):
        colors = {
            0: (0, 0, 255), # Background: Blue
            1: (0, 255, 0), # image: Red
            2: (255, 0, 0)  # Text: Green
        }
        overlay = np.zeros_like(image)
        for cls, color in colors.items():
            overlay[mask == cls] = color
        combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return combined