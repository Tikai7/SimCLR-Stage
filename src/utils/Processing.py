import cv2
import numpy as np
import concurrent.futures
from PIL import Image
import random


class RandomRotation90:

    """
        Class that apply a random 90 degree rotation to an image.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        angle = random.choice([90, -90])
        return img.rotate(angle, expand=True)
    
class Processing:
    """
    Class that contains methods to apply different image processing
    techniques to an image.

    Methods
    -------
    apply_rotogravure_effect :  Apply a rotogravure effect to an image.
    """

    @staticmethod
    def apply_rotogravure_effect(
        image : Image, 
        dot_size : int = 2, 
        intensity : int = 128, 
        method : str = "dot", 
        alpha : float  = 0.5
    ) -> Image:
        """
        Apply a rotogravure effect to an image.

        Args
        ---------
        image : PIL.Image
            The image to apply the effect to.
        dot_size : int
            The size of the dots to use in the effect. (only for "dot" method)
        intensity : int
            The intensity of the effect. (0-255) (only for "dot" method)
        method : str
            The method to use to apply the effect. Can be "grid" or "dot".
        alpha : float
            The alpha value to use when applying the effect. (only for "grid" method)

        Returns
        -------
        PIL.Image
            The image with the rotogravure effect applied.

        """

        image = np.array(image)
        rotogravure_effect = None
        if method == "dot":
            normalized_image = image / 255.0

            dot_pattern = np.zeros((dot_size, dot_size), dtype=np.float32)
            cv2.circle(dot_pattern, (dot_size // 2, dot_size // 2), dot_size // 4, 1, -1)

            tiled_pattern = np.tile(dot_pattern, (image.shape[0] // dot_size + 1, image.shape[1] // dot_size + 1))
            tiled_pattern = tiled_pattern[:image.shape[0], :image.shape[1]]
            pattern_intensity = intensity / 255.0

            rotogravure_effect = normalized_image * (1 - pattern_intensity) + tiled_pattern * pattern_intensity
            rotogravure_effect = (rotogravure_effect * 255).astype(np.uint8)

        elif method == "grid":
            hatch = np.zeros_like(image)
            hatch[::2, :] = 255  
            hatch[:, ::2] = 255 
            rotogravure_effect = cv2.addWeighted(image, alpha, hatch, (1-alpha), 0)
        else:
            raise ValueError("Invalid method. Method should be one of ['grid', 'dot']")
        
        return Image.fromarray(rotogravure_effect)