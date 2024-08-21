import cv2
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import concurrent.futures


class SIFTDetector:
    """
    Class that contains methods to apply the SIFT algorithm to an image.

    Methods
    -------
    computeSIFT :  Compute the SIFT keypoints and descriptors of an image.
    getBestMatch :  Get the best match between the SIFT descriptors of two images.
    saveSIFTDescriptors :  Save the SIFT keypoints and descriptors of a list of images.
    loadSIFTDescriptors :  Load the SIFT keypoints and descriptors of a list of images.
    displayKeypoints :  Display the keypoints of two images.
    """



    @staticmethod
    def computeSIFT(image : cv2.imread) -> tuple:
        """
        Compute the SIFT keypoints and descriptors of an image.
        Args
        ---------
        image : cv2.imread
            The image to compute the SIFT keypoints and descriptors of.
        Returns
        -------
        tuple
            The keypoints and descriptors of the image.
        """
        try:
            sift_detector = cv2.SIFT_create()
            kp, des = sift_detector.detectAndCompute(image, None)
            return kp, des
        except Exception as e:
            print(e)
            return None, None
        
    @staticmethod
    def getBestMatch(image_query_des : list , liste_descriptor : list) -> list:
        """
        Get the best match between the SIFT descriptors of two images.
        Args
        ---------
        image_query_des : list
            The descriptors of the query image.
        liste_descriptor : list
            The list of descriptors of the images to compare to.
        Returns
        -------
        list
            The best match between the SIFT descriptors of the two images.
        """
    

        # Function to compute the match between the query image and the other images
        def compute_match(i, des):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(image_query_des, des, k=2)
            good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
            return i, len(good), good

        all_matches = []

        # Use ThreadPoolExecutor to parallelize the computation of the matches        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(compute_match, i, des): i for i, des in enumerate(liste_descriptor)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(liste_descriptor)):
                try:
                    result = future.result()
                    all_matches.append(result)
                except Exception as e:
                    print(f"Exception occurred for descriptor {futures[future]}: {e}")

        best_match = sorted(all_matches, key=lambda x: x[1], reverse=True)
        return best_match

    @staticmethod
    def saveSIFTDescriptors(all_images_sift : list, path:str):
        """
        Save the SIFT keypoints and descriptors of a list of images.
        Args
        ---------
        all_images_sift : list
            The list of keypoints and descriptors of the images.
        path : str
            The path to save the keypoints and descriptors to.
        """

        for i, image_sift in enumerate(all_images_sift):
            try:
                kp, des = image_sift
                kp_tuples = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
                with open(f"{path}/kp_{i}", 'wb') as f:
                    pickle.dump(kp_tuples, f)
                with open(f"{path}/des_{i}", 'wb') as f:
                    pickle.dump(des, f)
            except:
                continue

    def loadSIFTDescriptors(path : str, num_sift : int) -> list:

        """
        Load the SIFT keypoints and descriptors of a list of images.
        Args
        ---------
        path : str
            The path to load the keypoints and descriptors from.
        num_sift : int
            The number of SIFT descriptors to load.
        Returns
        -------
        list
            The list of keypoints and descriptors of the images.
        """

        all_images_sift = []
        for i in tqdm(range(num_sift)):
            try: 
                with open(f"{path}/kp_{i}", 'rb') as f:
                    kp_tuples = pickle.load(f)
                
                kp = [
                    cv2.KeyPoint(
                        x=pt[0][0], y=pt[0][1], size=pt[1], 
                        angle=pt[2], response=pt[3], octave=pt[4], class_id=pt[5]
                    ) 
                    for pt in kp_tuples
                ]

                with open(f"{path}/des_{i}", 'rb') as f:
                    des = pickle.load(f)
                all_images_sift.append((kp, des))   
            except Exception as e:
                continue
        return all_images_sift

    @staticmethod
    def displayKeypoints(
        image_query : cv2.imread, 
        image_kp : list, 
        image_best : cv2.imread, 
        image_best_kp : list, 
        good_kp : list
    ):

        """
        Display the keypoints of two images.
        Args
        ---------
        image_query : cv2.imread
            The query image.
        image_kp : list
            The keypoints of the query image.
        image_best : cv2.imread
            The best image.
        image_best_kp : list
            The keypoints of the best image.
        good_kp : list
            The best matches between the keypoints of the two images.
        """

        image_result = cv2.drawMatchesKnn(image_query,image_kp,image_best,image_best_kp, good_kp, None, flags=2)
        plt.figure(figsize=(12,7))
        plt.imshow(image_result)
        plt.title("Keypoints of the two images")
        plt.show()