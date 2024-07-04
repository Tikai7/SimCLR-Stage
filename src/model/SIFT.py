import cv2
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
class SIFTDetector:
    @staticmethod
    def computeSIFT(image):
        try:
            sift_detector = cv2.SIFT_create()
            kp, des = sift_detector.detectAndCompute(image, None)
            return kp, des
        except:
            return None, None
    @staticmethod
    def getBestMatch(image_query_des, liste_descriptor):
        all_matches = []
        bf = cv2.BFMatcher()
        for i,des in tqdm(enumerate(liste_descriptor)):
            try:
                matches = bf.knnMatch(image_query_des,des, k=2)
                good = [[m] for m,n in matches if m.distance < 0.75*n.distance]
                all_matches.append((i,len(good),good))
            except Exception as e:
                print(e)
                continue

        best_match = sorted(all_matches, key=lambda x: x[1], reverse=True)

        return best_match

    @staticmethod
    def saveSIFTDescriptors(all_images_sift, path):
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

    def loadSIFTDescriptors(path, num_sift):
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
    def displayKeypoints(image_query, image_kp, image_best, image_best_kp, good_kp):
        image_result = cv2.drawMatchesKnn(image_query,image_kp,image_best,image_best_kp, good_kp, None, flags=2)
        plt.figure(figsize=(12,7))
        plt.imshow(image_result)
        plt.title("Keypoints of the two images")
        plt.show()