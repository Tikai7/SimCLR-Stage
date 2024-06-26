import cv2
import matplotlib.pyplot as plt

class SIFTDetector:
    @staticmethod
    def computeSIFT(image):
        sift_detector = cv2.SIFT_create()
        kp, des = sift_detector.detectAndCompute(image, None)
        return kp,des
    
    @staticmethod
    def getBestMatch(image_query_des, liste_descriptor):
        all_matches = []
        bf = cv2.BFMatcher()
        for i,des in enumerate(liste_descriptor):
            matches = bf.knnMatch(image_query_des,des, k=2)
            good = [[m] for m,n in matches if m.distance < 0.75*n.distance]
            all_matches.append((i,len(good),good))

        best_match = max(all_matches, key=lambda x: x[1])
        return best_match

    @staticmethod
    def saveSIFTDescriptors(all_images_sift, path):
        for i, image_sift in enumerate(all_images_sift):
            kp, des = image_sift
            with open(f"{path}/sift_{i}.txt", "w") as f:
                f.write(f"{kp}\n")
                f.write(f"{des}\n")

    @staticmethod
    def loadSIFTDescriptors(path):
        all_images_sift = []
        for i in range(1000):
            with open(f"{path}/sift_{i}.txt", "r") as f:
                kp = f.readline()
                des = f.readline()
                all_images_sift.append((kp,des))
        return all_images_sift

    @staticmethod
    def displayKeypoints(image_query, image_kp, image_best, image_best_kp, good_kp):
        image_result = cv2.drawMatchesKnn(image_query,image_kp,image_best,image_best_kp, good_kp, None, flags=2)
        plt.figure(figsize=(12,7))
        plt.imshow(image_result)
        plt.title("Keypoints of the two images")
        plt.show()