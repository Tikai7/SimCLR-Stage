import cv2
class ImageExtractor:

    @staticmethod
    def extract(image):
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

    